from PIL import Image

def ensure_rgb(img: Image.Image, bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Convert PIL image to RGB. Handles P mode directly, LA/RGBA by pasting on background.
    """
    if img.mode == "P":
        img = img.convert("RGBA")
    
    if img.mode in ("LA", "RGBA"):
        bg = Image.new("RGB", img.size, bg_color)
        # Handle LA correctly: convert to RGBA first if needed or just split
        if img.mode == "LA":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.getchannel("A"))
        return bg
        
    return img if img.mode == "RGB" else img.convert("RGB")
