import glob
import json

for f in sorted(glob.glob("notebooks/executed_*.ipynb")):
    nb = json.load(open(f, encoding="utf-8"))
    errs = [
        o.get("ename", "")
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    streams = [
        "".join(o.get("text", ""))
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    ]
    print(f.split("\\")[-1], "errors:", errs if errs else "NONE")
    for s in streams[:2]:
        print("   |", s[:300].replace("\n", " / "))
