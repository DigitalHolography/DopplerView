# installer/DopplerView.spec
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Root of the project (one level up from installer/)
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = []
datas += collect_data_files("dopplerview")
datas += collect_data_files("tkinterdnd2")
datas += collect_data_files("onnxruntime")
datas += collect_data_files("huggingface_hub")
datas += collect_data_files("scipy")
datas += collect_data_files("skimage")

# metadata (important for lazy loaders)
datas += copy_metadata("imageio")
datas += copy_metadata("tqdm")
datas += copy_metadata("scikit-image")
datas += copy_metadata("scipy")
datas += copy_metadata("numpy")
datas += copy_metadata("torch")
datas += copy_metadata("onnxruntime")

hiddenimports = [
    # tkinter
    "tkinterdnd2",
    # scipy — force eager loading to fix _distn_infrastructure NameError
    "scipy",
    "scipy.stats",
    "scipy.stats._distn_infrastructure",
    "scipy.stats._stats_py",
    "scipy.stats.distributions",
    "scipy.special",
    "scipy.special._ufuncs",
    "scipy.special._cython_special",
    "scipy.linalg",
    "scipy.linalg.blas",
    "scipy.linalg.lapack",
    "scipy.linalg._decomp_update",
    "scipy.sparse.csgraph",
    "scipy.sparse.linalg",
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy.linalg',
    # skimage — force-import modules loaded lazily via lazy_loader
    "skimage",
    "skimage.filters",
    "skimage.filters.ridges",
    "skimage.feature",
    "skimage.feature.corner",
    "skimage.morphology",
    "skimage.segmentation",
    "skimage.measure",
    "skimage.transform",
    # sklearn
    "sklearn",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
    # cv2
    "cv2",
    # ML
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "torch",
    "torchvision",
    # huggingface
    "huggingface_hub",
    "huggingface_hub.utils",
    # misc
    "PIL",
    "PIL.Image",
    "h5py",
    "pandas",
    "yaml",
    "tqdm",
]

hiddenimports += collect_submodules("scipy.stats")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("onnxruntime")

a = Analysis(
    [os.path.join(ROOT, "dopplerview", "tk_app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas + [
        (os.path.join(ROOT, "dopplerview", "resources"), "dopplerview/resources"),
        (os.path.join(ROOT, "DopplerView.ico"), "."),
    ],
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPECPATH)],   # hooks are in installer/ alongside the spec
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DopplerView",
    console=True,
    icon=os.path.join(ROOT, "DopplerView.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DopplerView",
)