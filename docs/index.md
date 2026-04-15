# ComplexHeatmap (Python)

A Python port of the R [ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) package for creating complex heatmap visualizations, using **grid_py** as the rendering backend.

## Features

- **Single heatmaps** with flexible clustering, splitting, and annotations
- **Heatmap lists** — compose multiple heatmaps and annotations side by side
- **20+ annotation types** — barplots, boxplots, points, lines, density, text, images, and more
- **UpSet plots** — visualize set intersections from combination matrices
- **OncoPrint** — visualize genomic alteration landscapes
- **Density/frequency heatmaps** — KDE and histogram-based visualizations
- **3D heatmaps** — oblique-projection 3D visualizations
- **Decoration system** — modify drawn heatmaps post-rendering
- **Legend system** — discrete and continuous legends with flexible packing

## Installation

```bash
pip install -e .
```

### Dependencies

- `numpy`, `scipy`, `pandas` — numerical computing
- `grid_py` — rendering backend (Python port of R's grid graphics)
- `scales` — color interpolation
- `Pillow` — image loading

## Quick Start

```python
import numpy as np
import complexheatmap as ch

# Create a simple heatmap
mat = np.random.randn(20, 10)
ht = ch.Heatmap(mat, name="example")
ht.draw(output="heatmap.png")

# Add annotations
ha = ch.HeatmapAnnotation(
    group=ch.anno_simple(["A"] * 5 + ["B"] * 5),
    score=ch.anno_barplot(np.random.rand(10)),
)
ht = ch.Heatmap(mat, name="annotated", top_annotation=ha)
ht.draw(output="annotated_heatmap.png")

# Compose multiple heatmaps
ht_list = ch.Heatmap(mat, name="ht1") + ch.Heatmap(mat, name="ht2")
ht_list.draw(output="heatmap_list.png")
```

## R Package Correspondence

This is a tutorial-scoped port of R ComplexHeatmap v2.25.3, covering 78 of 274 exports required by the official tutorials. The rendering backend is `grid_py` (a Python port of R's `grid` package), preserving the viewport-based layout model of the original.

| R | Python |
|---|--------|
| `Heatmap(mat)` | `ch.Heatmap(mat)` |
| `draw(ht)` | `ht.draw()` |
| `ht1 + ht2` | `ht1 + ht2` |
| `HeatmapAnnotation(...)` | `ch.HeatmapAnnotation(...)` |
| `oncoPrint(mat)` | `ch.onco_print(mat)` |
| `UpSet(m)` | `ch.UpSet(m)` |
