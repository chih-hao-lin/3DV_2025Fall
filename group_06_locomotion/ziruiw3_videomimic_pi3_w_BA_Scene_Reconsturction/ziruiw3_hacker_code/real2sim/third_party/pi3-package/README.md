# pi3-package

This is a lightweight wrapper to record Pi3 runtime dependencies for the VideoMimic pipeline.

Notes:
- The actual Pi3 source lives in `../../../../Pi3` relative to this folder. The reconstruction script appends both this folder and the Pi3 repo to `sys.path`.
- You will compile and set up any third-party components following the official Pi3 instructions.

Installation of Python deps (recommended in your existing env):

```bash
pip install -r requirements.txt
```

Expected layout:
- `VideoMimic/real2sim/stage1_reconstruction/pi3_reconstruction.py` uses `pi3` from the Pi3 repo.
- This folder is here only to declare and pin dependencies consistent with `Pi3/requirements.txt`.





