# ---- ultra-robust cudaFont.OverlayText adapter (handles many jetson-utils variants) ----
def _get_font_colors(font):
    white  = getattr(font, "White",  (255, 255, 255, 255))
    gray40 = getattr(font, "Gray40", (128, 128, 128, 255))
    return white, gray40

_OVERLAY_CALL = None  # cache the working variant index

def overlay_text(font, img, text, x=5, y=5,
                 fg=(255, 255, 255, 200), bg=(0, 0, 0, 80)):
    global _OVERLAY_CALL

    W = getattr(img, "width", None)
    H = getattr(img, "height", None)
    white, gray40 = _get_font_colors(font)

    if _OVERLAY_CALL is None:
        candidates = []
        if W is not None and H is not None:
            candidates += [
                lambda: font.OverlayText(img, W, H, text, x, y, white, gray40),
                lambda: font.OverlayText(img, W, H, text, x, y, fg, bg),
                lambda: font.OverlayText(img, W, H, text, x, y),
            ]
        candidates += [
            lambda: font.OverlayText(img, text, x, y, white, gray40),
            lambda: font.OverlayText(img, text, x, y, fg),
            lambda: font.OverlayText(img, x, y, text, fg),
            lambda: font.OverlayText(img, x, y, text, *fg),
            lambda: font.OverlayText(img, text, x, y, fg, bg),
            lambda: font.OverlayText(img, x, y, text, fg, bg),
        ]
        for i, call in enumerate(candidates):
            try:
                call(); _OVERLAY_CALL = i; return
            except Exception:
                continue
        try:
            out = globals().get("out")
            if out is not None:
                out.SetStatus(text); return
        except Exception:
            pass
        raise RuntimeError("cudaFont.OverlayText() signature not recognized on this build")

    if W is None or H is None:
        W = getattr(img, "width", 0); H = getattr(img, "height", 0)

    if _OVERLAY_CALL == 0:   return font.OverlayText(img, W, H, text, x, y, white, gray40)
    if _OVERLAY_CALL == 1:   return font.OverlayText(img, W, H, text, x, y, fg, bg)
    if _OVERLAY_CALL == 2:   return font.OverlayText(img, W, H, text, x, y)
    if _OVERLAY_CALL == 3:   return font.OverlayText(img, text, x, y, white, gray40)
    if _OVERLAY_CALL == 4:   return font.OverlayText(img, text, x, y, fg)
    if _OVERLAY_CALL == 5:   return font.OverlayText(img, x, y, text, fg)
    if _OVERLAY_CALL == 6:   return font.OverlayText(img, x, y, text, *fg)
    if _OVERLAY_CALL == 7:   return font.OverlayText(img, text, x, y, fg, bg)
    if _OVERLAY_CALL == 8:   return font.OverlayText(img, x, y, text, fg, bg)