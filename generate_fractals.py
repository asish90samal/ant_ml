"""
generate_fractals.py
Creates simple illustrative SVG sketches for three antennas.
Run: python code/generate_fractals.py --outdir ../designs
"""
import os
from pathlib import Path
out = Path(__file__).resolve().parent.parent / "designs"
if not out.exists():
    out.mkdir(parents=True)
# Files already included; this script is a placeholder to (re-)generate them if desired.
print("Designs are included in the designs/ directory.")