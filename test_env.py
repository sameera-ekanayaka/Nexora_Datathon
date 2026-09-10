import sys
import platform

libs = [
    "pandas", "numpy", "matplotlib", "seaborn",
    "scipy", "sklearn", "xgboost", "lightgbm",
    "plotly", "polars", "pyarrow", "duckdb"
]

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.system())

print("\n--- Import check ---")
for lib in libs:
    try:
        mod = __import__(lib)
        version = getattr(mod, "__version__", "unknown")
        print(f"✅ {lib:12s} OK (version {version})")
    except ImportError as e:
        print(f"❌ {lib:12s} FAILED: {e}")
        if lib == "xgboost":
            system = platform.system()
            if system == "Darwin":
                print("   → Fix (macOS): run `brew install libomp`, then retry.")
            elif system == "Windows":
                print("   → Fix (Windows): install the Microsoft Visual C++ Redistributable (x64):")
                print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
            elif system == "Linux":
                print("   → Fix (Linux): run `sudo apt-get install libgomp1` (Debian/Ubuntu) or equivalent.")