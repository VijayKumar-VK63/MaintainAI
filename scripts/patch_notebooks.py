import json

BOOT = [
    "import os, sys\n",
    "from pathlib import Path\n",
    "ROOT = Path().resolve().parent if Path().resolve().name == 'notebooks' else Path().resolve()\n",
    "os.chdir(ROOT)\n",
    "sys.path.insert(0, str(ROOT))\n",
]
FILES = [
    "notebooks/01_environment_check.ipynb",
    "notebooks/02_dataset_analysis.ipynb",
    "notebooks/03_preprocessing.ipynb",
    "notebooks/04_predictive_model.ipynb",
    "notebooks/05_slm_dataset_creation.ipynb",
]
for f in FILES:
    nb = json.load(open(f, encoding="utf-8"))
    first_src = "".join(nb["cells"][0].get("source", []))
    cell = {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": BOOT,
    }
    if first_src.startswith("import sys") or first_src.startswith("import os, sys"):
        nb["cells"][0] = cell
        print("replaced bootstrap", f)
    else:
        nb["cells"].insert(0, cell)
        print("inserted bootstrap", f)
    json.dump(nb, open(f, "w", encoding="utf-8"), indent=1)
