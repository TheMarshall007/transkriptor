# PyInstaller onedir build for the local desktop application.
from PyInstaller.utils.hooks import collect_all

whisper_datas, whisper_binaries, whisper_hiddenimports = collect_all("whisper")

a = Analysis(
    ["transkriptor_gui.py"],
    pathex=[],
    binaries=whisper_binaries,
    datas=whisper_datas,
    hiddenimports=whisper_hiddenimports + ["pynvml"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Transkriptor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
collect = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Transkriptor",
)
