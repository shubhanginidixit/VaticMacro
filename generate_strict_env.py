"""
generate_strict_env.py
Freeze the exact working environment into requirements-strict.txt.

Iterates over every critical library used by VaticMacro, reads its installed
version, and writes a pinned (==) requirements file.  Libraries that are not
installed are skipped with a warning so the script never crashes.
"""

# Map of pip-install name -> import name.
# Most packages share the same name, but scikit-learn is imported as sklearn.
CRITICAL_LIBRARIES = {
    "flask": "flask",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}

OUTPUT_FILE = "requirements-strict.txt"


def main():
    pinned_lines = []

    for pip_name, import_name in CRITICAL_LIBRARIES.items():
        try:
            module = __import__(import_name)
            version = module.__version__
            pinned_lines.append(f"{pip_name}=={version}")
        except ImportError:
            print(f"WARNING: '{pip_name}' (import as '{import_name}') is not installed — skipping.")
        except AttributeError:
            # Very unlikely, but guard against modules without __version__
            print(f"WARNING: '{pip_name}' has no __version__ attribute — skipping.")

    if not pinned_lines:
        print("ERROR: No critical libraries found. Nothing written.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(pinned_lines) + "\n")

    print(f"Wrote {len(pinned_lines)} pinned dependencies to {OUTPUT_FILE}:")
    for line in pinned_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
