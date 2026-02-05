from pathlib import Path
import pettingzoo
envs = Path(pettingzoo.__path__[0]).glob("**/*_v?.py")
keys = []
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    keys.append(f"pz-{lib}-{name}")
for k in sorted(keys):
    print(k)