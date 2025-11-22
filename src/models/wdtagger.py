# src/models/wdtagger.py
import io, csv, gzip, torch
from PIL import Image
from huggingface_hub import hf_hub_download
import timm
from timm.data import resolve_model_data_config, create_transform
from src.eval.util import dtype_from_str, img_rgba_to_rgb

def _load_tag_list(repo_id: str):
	candidates = ["selected_tags.csv", "selected_tags.csv.gz"]
	last_err = None
	for fname in candidates:
		try:
			path = hf_hub_download(repo_id=repo_id, filename=fname)
			if fname.endswith(".gz"):
				with gzip.open(path, "rt", encoding="utf-8") as f:
					text = f.read()
			else:
				with open(path, "r", encoding="utf-8") as f:
					text = f.read()

			buf = io.StringIO(text)
			reader = csv.DictReader(buf)
			if reader.fieldnames:
				fns_lower = [fn.lower() for fn in reader.fieldnames]
				if "name" in fns_lower:
					name_key = reader.fieldnames[fns_lower.index("name")]
					return [row[name_key] for row in reader]

			buf.seek(0)
			rows = list(csv.reader(buf))
			header_lower = [h.lower() for h in rows[0]]
			name_idx = header_lower.index("name") if "name" in header_lower else 1
			return [r[name_idx] for r in rows[1:]]
		except Exception as e:
			last_err = e
			continue
	raise RuntimeError(f"Could not resolve tag list from {candidates}. Last error: {last_err}")


def _build_rating_index(tags):
	idx = {}
	for i, t in enumerate(tags):
		if isinstance(t, str) and t.startswith("rating:"):
			idx[t.split("rating:")[1]] = i

	for name in ["general", "sensitive", "questionable", "explicit"]:
		if name not in idx:
			try:
				idx[name] = tags.index(f"rating:{name}")
			except ValueError:
				try:
					idx[name] = tags.index(name)
				except ValueError:
					pass

	missing = [k for k in ["general", "sensitive", "questionable", "explicit"] if k not in idx]
	if missing:
		print("[wdtagger] WARNING: rating tags not found; "
		      "falling back to [0,1,2,3]=[general,sensitive,questionable,explicit].")
		idx = {"general":0, "sensitive":1, "questionable":2, "explicit":3}
	return idx


class WDEva02TaggerZS:

	def __init__(self,
	             repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
	             dtype="float16"):
		self.repo_id = repo_id
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.torch_dtype = dtype_from_str(dtype)

		self.model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).to(self.device).eval()
		self.model = self.model.to(self.torch_dtype)
		self.model = self.model.to(memory_format=torch.channels_last)

		cfg = resolve_model_data_config(self.model)
		self.preprocess = create_transform(**cfg)

		self.tags = _load_tag_list(repo_id)
		self.idx_rating = _build_rating_index(self.tags)

		self._cg = None
		self._static_x = None
		self._static_probs = None
		self._cg_batch = None

	# --------- helpers ----------

	def _encode(self, pil_img: Image.Image):
		pil_img = img_rgba_to_rgb(pil_img)
		x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
		param_dtype = next(self.model.parameters()).dtype
		return x.to(param_dtype)

	@torch.inference_mode()
	def predict_proba(self, pil_img, category: str) -> float:
		x = self._encode(pil_img)
		param_dtype = next(self.model.parameters()).dtype
		with torch.autocast(device_type="cuda", dtype=param_dtype, enabled=True):
			logits = self.model(x)
		probs = torch.sigmoid(logits)[0]

		p_exp  = float(probs[self.idx_rating["explicit"]])
		p_ques = float(probs[self.idx_rating["questionable"]])
		p_sens = float(probs[self.idx_rating["sensitive"]])
		p_unsafe = max(p_exp, p_ques, p_sens)
		return float(p_unsafe)

	@torch.inference_mode()
	def predict_proba_batch(self, pil_list, categories=None):
		param_dtype = next(self.model.parameters()).dtype

		xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
		x = torch.stack(xs, dim=0)
		x = x.pin_memory().to(
			device=self.device,
			dtype=param_dtype,
			non_blocking=True,
			memory_format=torch.channels_last
		)
		B = x.shape[0]
		need_capture = (self._cg is None) or (self._cg_batch != B)

		if need_capture:
			torch.cuda.synchronize()
			self._cg = torch.cuda.CUDAGraph()
			self._cg_batch = B
			self._static_x = torch.empty_like(x, memory_format=torch.channels_last)
			self._static_probs = torch.empty((B, len(self.tags)),
			                                 device=self.device, dtype=torch.float16)
			for _ in range(2):
				_ = self.model(self._static_x)

			with torch.cuda.graph(self._cg):
				logits = self.model(self._static_x)
				probs  = torch.sigmoid(logits)
				self._static_probs.copy_(probs.to(self._static_probs.dtype))

		self._static_x.copy_(x, non_blocking=True)
		self._cg.replay()
		probs = self._static_probs

		ie = self.idx_rating["explicit"]
		iq = self.idx_rating["questionable"]
		is_ = self.idx_rating["sensitive"]
		unsafe = torch.maximum(probs[:, ie], torch.maximum(probs[:, iq], probs[:, is_]))

		return unsafe.float().cpu().tolist()



