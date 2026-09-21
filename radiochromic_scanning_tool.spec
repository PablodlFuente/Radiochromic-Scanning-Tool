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
splash = Splash(
    "resources/radiochromic_film_analyzer.png",
    binaries=analysis.binaries,
    datas=analysis.datas,
    text_pos=(20, 360),
    text_size=12,
    text_color="#2f4f4f",
    text_default="Starting Radiochromic Film Analyzer…",
    center="active",
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    splash.binaries,
    splash,
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
