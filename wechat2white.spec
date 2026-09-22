# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['pdf_bg_to_white.py'],
    pathex=[],
    binaries=[],
    datas=[('/new/test/yolo_pdf_bg2white/best.onnx', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Runtime only needs ONNX Runtime; training/export frameworks and their
    # optional analysis dependencies must not be pulled into the executable.
    excludes=[
        'onnx',
        'ultralytics',
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'scipy',
        'sympy',
        'matplotlib',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='wechat2white',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
