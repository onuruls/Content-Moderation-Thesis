import time, sys, shutil

def _fmt_hms(secs: float) -> str:
	secs = max(0, int(secs))
	h = secs // 3600
	m = (secs % 3600) // 60
	s = secs % 60
	return f"{h:02d}:{m:02d}:{s:02d}"

class ProgressBar:
	def __init__(self, total: int, desc: str = "", width: int | None = None, min_interval: float = 0.1):
		self.total = max(1, int(total))
		self.desc = desc
		self.min_interval = float(min_interval)
		self.start = time.time()
		self.last_print = 0.0
		self.seen = 0
		self.is_tty = sys.stdout.isatty()
		try:
			termw = shutil.get_terminal_size((100, 20)).columns
		except Exception:
			termw = 100
		self.width = int(width) if width else max(10, min(40, termw - 60))

	def update(self, inc: int = 1):
		self.seen = min(self.total, self.seen + int(inc))
		now = time.time()
		if (now - self.last_print) >= self.min_interval or self.seen == self.total:
			self._print(now)

	def set(self, n: int):
		self.seen = min(self.total, max(0, int(n)))
		now = time.time()
		if (now - self.last_print) >= self.min_interval or self.seen == self.total:
			self._print(now)

	def _print(self, now: float):
		elapsed = now - self.start
		rate = self.seen / elapsed if elapsed > 0 else 0.0
		remain = (self.total - self.seen) / rate if rate > 0 else 0.0
		pct = self.seen / self.total
		filled = int(self.width * pct)
		bar = "#" * filled + "-" * (self.width - filled)
		msg = (f"{self.desc} [{bar}] {pct*100:5.1f}%  "
		       f"{self.seen}/{self.total}  "
		       f"{rate:5.1f} it/s  "
		       f"elapsed { _fmt_hms(elapsed) }  eta { _fmt_hms(remain) }")
		if self.is_tty:
			sys.stdout.write("\r" + msg)
			if self.seen == self.total:
				sys.stdout.write("\n")
			sys.stdout.flush()
		else:
			if self.seen == self.total:
				print(msg)
			elif now - self.last_print >= max(self.min_interval, 2.0):
				print(msg)
		self.last_print = now

	def close(self):
		if self.is_tty and self.seen < self.total:
			self.set(self.total)
