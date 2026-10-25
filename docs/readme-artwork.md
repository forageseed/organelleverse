# README artwork

The header preserves the existing blue-V/DNA/green-and-orange OrganelleVerse logo on the left
and displays eighteen separately editable tool illustrations on the right.
The white canvas, forest-green lines, sage fills and warm gold/orange accents
follow the existing brand. The tool illustrations are original SVG geometry.

## Files and assembly

- `assets/tool-icons/*.svg`: one independent, titled 64 × 64 SVG per tool family.
- `assets/readme-tools.json`: labels, source modules and icon order.
- `assets/logo.png`: the existing GitHub logo, unchanged; this is the single
  brand source read by the builder, not a similarly named legacy logo.
- `assets/readme-logo.svg`: embeds that PNG and crops its outer whitespace with
  an SVG viewBox. The logo remains a raster original; it is not a vector trace.
- `assets/ov-readme-header.svg`: self-contained 1280 × 430 composite.
- `assets/ov-tool-icons.svg`: tool-icon contact sheet.

Rebuild with the project's Python interpreter:

```bash
python scripts/build_readme_artwork.py
```

The builder uses only the standard library. It inlines icon geometry and the
logo, so the composed SVG does not require JavaScript, foreignObject, external
fonts, remote images or nested file fetches to display as a README image.

```html
<img src="docs/assets/ov-readme-header.svg"
     alt="OrganelleVerse logo and scientific toolkit"
     width="1280">
```

The manifest groups the actual scientific modules into readable families.
An illustration is a navigation summary, not a claim that every method, model
or optional executable is installed. Assembly, annotation, comparison, trees,
population/variation, RNA, proteins, morphology and visualization refer to OV
scientific functions. Python and agent icons represent two interfaces to the
same scientific core. Desktop sessions, connector management and compute
scheduling are outside this artwork's OV feature claims.

## Design brief

Create a compact GitHub README header for OrganelleVerse. Preserve the current
GitHub `docs/assets/logo.png` (blue V and DNA, green chloroplast, orange mitochondrion)
on the left, cropping only
the empty outer canvas. On the right, arrange the eighteen scientific tool
families in a clear three-row grid. Draw each family as a separate 64 × 64 SVG
with rounded 2.2-unit forest-green strokes, pale sage and cream fills, and small
gold/orange accents. Use distinct scientific metaphors rather than repeating
one generic icon. Keep short English labels, a white background, a restrained
botanical research-notebook mood and a compact serif wordmark. Assemble the
original logo and the actual SVG geometry into one self-contained banner.
Avoid screenshots, large blank margins, heavy card borders, decorative fake
metrics, host-specific AI logos, and claims about unverified scientific results.
