# hook-scipy.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from pathlib import Path
import scipy.stats._distn_infrastructure as _mod
import shutil

hiddenimports = collect_submodules('scipy.stats')
hiddenimports += collect_submodules('scipy.special')
hiddenimports += collect_submodules('scipy._lib')
hiddenimports += collect_submodules('scipy._lib.array_api_compat')
hiddenimports += collect_submodules('scipy._lib.array_api_compat.numpy')
hiddenimports += [
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy.linalg',
]

datas = collect_data_files('scipy')

# Patch the file before PyInstaller bundles it
TARGET = Path(_mod.__file__)
BAD_LINE = "del obj"
FIXED_LINE = "# del obj  # Commented out: causes NameError in PyInstaller frozen exe"

text = TARGET.read_text(encoding="utf-8")
if BAD_LINE in text:
    TARGET.write_text(text.replace(BAD_LINE, FIXED_LINE, 1), encoding="utf-8")
    print(f"  [hook-scipy] Patched {TARGET}")
else:
    print(f"  [hook-scipy] No patch needed in {TARGET}")