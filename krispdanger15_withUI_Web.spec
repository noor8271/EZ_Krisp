# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['audioforshippingwithUI_web.py'],
    pathex=[],
    binaries=[],
    datas=[('DeepFilterNet3', 'DeepFilterNet3')],
    hiddenimports=['torch', 'df', 'df.enhance', 'df.deepfilternet3'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='krispdanger15_withUI_Web',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
