# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


a = Analysis(
    ['tools/gui_entry.py'],
    pathex=['src'],
    binaries=[],
    datas=[('assets/sync2act.ico', 'assets')],
    hiddenimports=sorted(
        collect_submodules('huggingface_hub')
        + ['av', 'pyarrow', 'pyarrow.parquet', 'tqdm.auto']
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pytest', 'sphinx', 'IPython', 'nbformat', 'jupyter', 'zmq', 'tkinter', 'cryptography', 'pyqtgraph.opengl', 'torch.utils.tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sync2Act',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/sync2act.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Sync2Act',
)
