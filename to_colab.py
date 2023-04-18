import jupytext

files = [
    "utils.py",  # utilities (run first)
    "google_streetview_query.py",
    "pit_orl_manh.py",
    "model.py",
    "main.py",
]

nb = None
for f in files:
    print(f"Reading input py file: \"{f}\"")
    if nb is None:
        nb = jupytext.read(f, fms="py")
    else:
        nb.cells.extend(jupytext.read(f, fms="py").cells)
jupytext.write(nb, "geo-estimator.ipynb", fmt=".ipynb")
