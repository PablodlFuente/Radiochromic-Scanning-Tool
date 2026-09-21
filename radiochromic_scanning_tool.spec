from PyInstaller.utils.hooks import collect_submodules


hidden_imports = collect_submodules("custom_plugins")

analysis = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("images", "images"), ("docs", "docs"), ("resources", "resources")],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="RadiochromicFilmAnalyzer",
    icon="resources/radiochromic_film_analyzer.ico",
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
)
