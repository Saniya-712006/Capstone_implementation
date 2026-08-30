"""Shared helpers for hand-authored architecture-diagram SVGs (standalone files, print/report styling -- fixed colors, not currentColor theming, since these embed into a static document)."""

FONT = "font-family=\"Segoe UI, Helvetica, Arial, sans-serif\""

COLORS = {
    "physics": ("#fed7aa", "#c2410c"),   # PhysNet -- orange
    "chem":    ("#bfdbfe", "#1d4ed8"),   # ChemNet -- blue
    "data":    ("#e5e7eb", "#4b5563"),   # data / infra -- gray
    "output":  ("#bbf7d0", "#15803d"),   # readout / output -- green
    "cal":     ("#e9d5ff", "#7e22ce"),   # CAL head -- purple
    "explain": ("#fecaca", "#b91c1c"),   # Phase-3 explainer -- red
    "key":     ("#fde68a", "#b45309"),   # the one mechanism under discussion -- amber
}


def svg_open(w, h, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{title}">\n'
        f'<title>{title}</title>\n'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'<defs>\n'
        f'  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n'
        f'    <path d="M0,0 L10,5 L0,10 z" fill="#374151"/>\n'
        f'  </marker>\n'
        f'  <marker id="arrowkey" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n'
        f'    <path d="M0,0 L10,5 L0,10 z" fill="#b45309"/>\n'
        f'  </marker>\n'
        f'</defs>\n'
    )


def svg_close():
    return "</svg>\n"


def box(x, y, w, h, label, sub=None, kind="data", rx=8, dashed=False):
    """`sub` may contain \\n for a multi-line subtitle -- split and stack manually, since SVG <text> ignores \\n."""
    fill, stroke = COLORS[kind]
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>']
    cx = x + w / 2
    if sub:
        sub_lines = sub.split("\n")
        label_y = y + h/2 - 4 - (len(sub_lines) - 1) * 6.5
        out.append(f'<text x="{cx}" y="{label_y}" text-anchor="middle" {FONT} font-size="12.5" font-weight="600" fill="#1f2937">{label}</text>')
        for i, line in enumerate(sub_lines):
            out.append(f'<text x="{cx}" y="{label_y + 16 + i*13}" text-anchor="middle" {FONT} font-size="10.5" fill="#4b5563">{line}</text>')
    else:
        out.append(f'<text x="{cx}" y="{y + h/2 + 4}" text-anchor="middle" {FONT} font-size="12.5" font-weight="600" fill="#1f2937">{label}</text>')
    return "\n".join(out)


def group(x, y, w, h, label, kind="data"):
    """A large dashed enclosing boundary representing a module/stage grouping, label at top-left."""
    _, stroke = COLORS[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" '
        f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="3,4"/>\n'
        f'<text x="{x + 10}" y="{y - 8}" {FONT} font-size="12" font-weight="700" fill="{stroke}">{label}</text>'
    )


def txt(x, y, content, size=12, anchor="middle", weight="400", color="#1f2937", italic=False, line_height=15):
    """SVG <text> does not honour \\n -- split multi-line content into stacked <text> elements ourselves."""
    style = ' font-style="italic"' if italic else ""
    lines = content.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(f'<text x="{x}" y="{y + i * line_height}" text-anchor="{anchor}" {FONT} font-size="{size}" font-weight="{weight}" fill="{color}"{style}>{line}</text>')
    return "\n".join(out)


def arrow(x1, y1, x2, y2, label=None, dashed=False, key=False, label_dx=0, label_dy=-6):
    stroke = "#b45309" if key else "#374151"
    marker = "arrowkey" if key else "arrow"
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>']
    if label:
        mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
        out.append(f'<text x="{mx}" y="{my}" text-anchor="middle" {FONT} font-size="10.5" fill="{stroke}">{label}</text>')
    return "\n".join(out)


def elbow(points, label=None, dashed=False, key=False, label_at=-1, label_dx=0, label_dy=-6):
    """Polyline through a list of (x,y) points, arrowhead at the end."""
    stroke = "#b45309" if key else "#374151"
    marker = "arrowkey" if key else "arrow"
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    out = [f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>']
    if label:
        x, y = points[label_at]
        px, py = points[label_at - 1]
        mx, my = (x + px) / 2 + label_dx, (y + py) / 2 + label_dy
        out.append(f'<text x="{mx}" y="{my}" text-anchor="middle" {FONT} font-size="10.5" fill="{stroke}">{label}</text>')
    return "\n".join(out)


def write_svg(path, w, h, title, body_parts, caption=None):
    parts = [svg_open(w, h, title)]
    parts.extend(body_parts)
    if caption:
        parts.append(txt(w / 2, h - 14, caption, size=11.5, color="#6b7280", italic=True))
    parts.append(svg_close())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", path)
