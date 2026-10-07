"""Run the official Graph WaveNet train.py unchanged, after patching Conv1d for current PyTorch.

Usage: python scripts/train_baseline.py [train.py args]
Default args reproduce the paper (README command), on cuda:0, with our data paths.
"""
import os
import runpy
import sys
from pathlib import Path

# Intel Fortran runtime (pulled in by numpy/scipy) kills the process on a console close event.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
ROOT = Path(__file__).resolve().parents[1]
if "--log" in sys.argv:  # for pythonw.exe, which has no console and no stdout
    i = sys.argv.index("--log")
    sys.stdout = sys.stderr = open(Path(sys.argv[i + 1]).resolve(), "a", buffering=1)
    del sys.argv[i: i + 2]
sys.path.insert(0, str(ROOT / "src"))
import gwn  # noqa: E402

gwn.patch_upstream()
defaults = ["--device", "cuda:0", "--data", str(ROOT / "data/METR-LA"),
            "--adjdata", str(ROOT / "data/sensor_graph/adj_mx.pkl"),
            "--gcn_bool", "--adjtype", "doubletransition", "--addaptadj", "--randomadj",
            "--save", str(ROOT / "checkpoints/metr")]
argv = defaults + sys.argv[1:]  # later flags win in argparse
for i, a in enumerate(argv[:-1]):
    if a == "--save":  # train.py runs from GWN_DIR, so make the prefix absolute
        argv[i + 1] = str(Path(argv[i + 1]).resolve())
        Path(argv[i + 1]).parent.mkdir(parents=True, exist_ok=True)
sys.argv = ["train.py"] + argv
os.chdir(gwn.GWN_DIR)
runpy.run_path("train.py", run_name="__main__")
