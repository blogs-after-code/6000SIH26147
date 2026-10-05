"""Build kaggle/train_modulation_cnn.ipynb from the cell-marked script (single source of truth)."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "train_modulation_cnn.py"), encoding="utf-8").read()
cells = []
for chunk in re.split(r"(?m)^# %%", src):
    if not chunk.strip():
        continue
    head, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        lines = [re.sub(r"^# ?", "", l) for l in body.splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines][:-1] + [lines[-1]]})
    else:
        lines = body.splitlines()
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in lines][:-1] + [lines[-1]]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "train_modulation_cnn.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out, len(cells), "cells")
