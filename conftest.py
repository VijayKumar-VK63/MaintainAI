import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
