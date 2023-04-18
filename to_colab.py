import jupytext

files = [
    "utils",  # utilities (run first)
    "gsv_query",
    "pit_orl_manh",
    "dataloader",
    "model",
    "main",
]

nb = None
for f in files:
    print(f'Reading input py file: "{f}"')
    if nb is None:
        nb = jupytext.read(f"{f}.py", fms="py")
    else:
        nb.cells.extend(jupytext.read(f"{f}.py", fms="py").cells)

# remove those cells that are local imports (won't work in notebook context
def remove_local_references(s: str) -> str:
    lines = s.split("\n")

    def matches(l: str) -> bool:
        raw_imports = [l == f"import {x}" for x in files]
        from_imports = [l.startswith(f"from {x} import") for x in files]
        return max(raw_imports) or max(from_imports)

    good_lines = [l for l in lines if not matches(l)]
    return "\n".join(good_lines)


# remove local references within this project (like from X import Y since in nb theyre all global anyways)
for c in nb.cells:
    c.source = remove_local_references(c.source)

# remove any empty cells (just for cleanliness)
non_empty_cells = []
for c in nb.cells:
    if len(c.source) > 0:
        non_empty_cells.append(c)
nb.cells = non_empty_cells

out: str = "geo-estimator.ipynb"
jupytext.write(nb, out, fmt=".ipynb")
print(f"Output .ipynb file to {out}")
