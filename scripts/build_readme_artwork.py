"""Compose the README header from the original logo and independent tool SVGs.

Standard library only. No scientific package imports, remote assets or network.
"""
from __future__ import annotations

import argparse
import base64
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def drawing(path: Path) -> str:
    """Inline native SVG geometry so an image viewer needs no external assets."""
    root = ET.parse(path).getroot()
    return ''.join(ET.tostring(child, encoding='unicode') for child in root
                   if child.tag == f'{{{NS}}}g')


def compose(root: Path) -> tuple[Path, Path]:
    assets = root / 'docs' / 'assets'
    tools = json.loads((assets / 'readme-tools.json').read_text(encoding='utf-8'))['tools']
    # Use the repository's existing GitHub brand asset as the single source.
    # Do not select similarly named legacy logos from other working copies.
    logo_data = base64.b64encode((assets / 'logo.png').read_bytes()).decode('ascii')
    (assets / 'readme-logo.svg').write_text(
        f'<svg xmlns="{NS}" width="300" height="300" viewBox="150 140 950 960" '
        'role="img" aria-labelledby="brand-title">\n'
        '<title id="brand-title">OrganelleVerse logo</title>\n'
        '<desc>The existing GitHub docs/assets/logo.png is preserved unchanged. '
        'The SVG viewBox crops only its surrounding whitespace.</desc>\n'
        f'<image x="0" y="0" width="1254" height="1254" href="data:image/png;base64,{logo_data}"/>\n'
        '</svg>\n', encoding='utf-8', newline='\n')
    logo = ET.parse(assets / 'readme-logo.svg').getroot()
    logo.attrib.update(x='30', y='96', width='300', height='292')
    brand = ET.tostring(logo, encoding='unicode')
    tiles = []
    sheet_tiles = []
    for index, tool in enumerate(tools):
        column, row = index % 6, index // 6
        center, top = 440 + 143 * column, 76 + 105 * row
        art = drawing(assets / tool['svg'])
        label = escape(tool['label'])
        description = escape(tool['description'])
        tint = '#f5eedc' if index % 3 == 1 else '#edf2e9'
        tile = f'''<g role="img" aria-label="{label}"><title>{label}: {description}</title>
<circle cx="{center}" cy="{top + 31}" r="32" fill="{tint}" opacity=".46"/>
<g transform="translate({center - 32} {top})">{art}</g>
<text x="{center}" y="{top + 86}" text-anchor="middle" font-size="14.2" font-weight="500">{label}</text>
</g>'''
        tiles.append(tile)
        center, top = 90 + column * 180, 38 + row * 146
        sheet_tiles.append(f'<g transform="translate({center-32} {top})">{art}</g>'
                           f'<text x="{center}" y="{top+92}" text-anchor="middle" font-size="15">{label}</text>')

    header = f'''<svg xmlns="{NS}" width="1280" height="430" viewBox="0 0 1280 430" role="img" aria-labelledby="ov-header-title ov-header-desc">
<title id="ov-header-title">OrganelleVerse — plant organelle science</title>
<desc id="ov-header-desc">The original OrganelleVerse logo on the left. Eighteen separate illustrations on the right represent assembly, annotation, retrieval, quality control, comparative genomics, phylogeny, pangenomes, selection, RNA, proteins, gene transfer, genome structure, variation, barcoding, morphology, visualization, Python and agent interfaces.</desc>
<rect width="1280" height="430" fill="#ffffff"/>
<g fill="#355b43" font-family="'Segoe UI', Arial, sans-serif">
<text x="34" y="58" font-family="Georgia, 'Times New Roman', serif" font-size="38" font-weight="700" letter-spacing="-1.4">OrganelleVerse</text>
<text x="36" y="86" font-size="14.5" fill="#6d7c69" letter-spacing=".25">Plant organelle science</text>
{brand}
<path d="M366 103v261" fill="none" stroke="#dce4d7" stroke-width="1.4" stroke-dasharray="3 7"/>
<text x="409" y="43" font-size="12.5" font-weight="600" letter-spacing="2.6" fill="#75816d">EXPLORE THE TOOLKIT</text>
{''.join(tiles)}
<text x="409" y="408" font-size="13.5" fill="#77816e">One scientific core. Python and agent interfaces.</text>
<path d="M59 402h184" stroke="#e5e7da" stroke-width="1.3" stroke-dasharray="2 6"/>
<path d="M286 398l3 7 7 3-7 3-3 7-3-7-7-3 7-3z" fill="none" stroke="#d9a33b" stroke-width="1.2"/>
</g>
</svg>
'''
    sheet = f'''<svg xmlns="{NS}" width="1080" height="464" viewBox="0 0 1080 464" role="img" aria-label="Eighteen individually editable OrganelleVerse tool icons">
<rect width="1080" height="464" fill="white"/>
<g fill="#355b43" font-family="'Segoe UI', Arial, sans-serif">{''.join(sheet_tiles)}</g>
</svg>
'''
    destinations = assets / 'ov-readme-header.svg', assets / 'ov-tool-icons.svg'
    for path, content in zip(destinations, (header, sheet), strict=True):
        path.write_text(content, encoding='utf-8', newline='\n')
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for path in compose(args.root.resolve()):
        print(path)


if __name__ == '__main__':
    main()
