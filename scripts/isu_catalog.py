import os
from dotenv import load_dotenv
from pathlib import Path

from openai import OpenAI

import graduate as grad

raw_catalog = Path("data/isu/raw.txt")
class_folder = Path("data/isu/classes")
csv_catalog = Path("data/isu/catalog.csv")
svg_catalog = Path("data/isu/catalog.svg")

dot = load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

grad.fBuildISUCourseCatalogCSV(class_folder, csv_catalog, raw_catalog, client)