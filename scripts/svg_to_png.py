"""Render an SVG file to PNG via headless Chromium (Playwright).

Usage: python scripts/svg_to_png.py <in.svg> <out.png> [scale]

The SVG is inlined into the page (more reliable than <img src=file://>),
and explicit width/height are taken from the SVG root element.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def svg_size(svg_text: str) -> tuple[int, int]:
    def attr(name: str) -> int | None:
        m = re.search(rf'{name}="([0-9.]+)(?:px)?"', svg_text[:2000])
        return int(float(m.group(1))) if m else None

    w, h = attr("width"), attr("height")
    if w and h:
        return w, h
    m = re.search(r'viewBox="([0-9.\s]+)"', svg_text[:2000])
    if m:
        parts = [float(x) for x in m.group(1).split()]
        return int(parts[2]), int(parts[3])
    raise SystemExit("cannot determine SVG size")


def main() -> None:
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    svg_text = src.read_text(encoding="utf-8")
    w, h = svg_size(svg_text)
    # strip XML prolog if present so the SVG inlines cleanly
    svg_inline = re.sub(r"<\?xml[^>]*\?>", "", svg_text, count=1)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:#fff}}</style></head>
<body><div id="fig" style="width:{w}px;height:{h}px">{svg_inline}</div></body></html>"""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h},
                                device_scale_factor=scale)
        page.set_content(html, wait_until="load")
        page.wait_for_timeout(400)
        page.locator("#fig").screenshot(path=str(dst))
        browser.close()
    print("png written:", dst, f"({w}x{h} @{scale}x)")


if __name__ == "__main__":
    main()
