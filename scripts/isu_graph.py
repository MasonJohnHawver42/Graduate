import os
from pathlib import Path
from dotenv import load_dotenv
from importlib.resources import files

import graduate as grad

csv_catalog = files("graduate.assets.isu").joinpath("catalog.csv")
svg_catalog = Path("data/isu/catalog.svg")

csv_catalog = grad.fLoadISUCourseCatalogCSV(csv_catalog)
catalog = grad.fBuildISUCourseCatalog(csv_catalog)

print(grad.fISUCatalogToStr(catalog))

graph = grad.fCatalogToGraph(catalog)
graph.render(svg_catalog.parent / svg_catalog.stem, format="svg", cleanup=True)
graph.render(svg_catalog.parent / svg_catalog.stem, format="png", cleanup=True)