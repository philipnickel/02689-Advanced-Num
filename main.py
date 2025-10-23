#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Hardcoded directories to scan for scripts ---
DIRS = ["assignment_2"]

# --- Where to collect figures afterwards ---
DEST_ROOT = Path("Overleaf_Report_A2/Figures")

MAX_CHARS = 4000  # cap error output


def discover_scripts() -> list[Path]:
    return sorted(
        p
        for d in DIRS
        for p in Path(d).rglob("*.py")
        if p.is_file() and p.name != Path(__file__).name
    )


def to_module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def run_script(path: Path):
    """Return (path, code, mode, stdout, stderr)."""
    r = subprocess.run(["python", str(path)], capture_output=True, text=True)
    if r.returncode == 0:
        return path, 0, "direct", r.stdout, r.stderr

    # fallback: as module
    modname = to_module_name(path)
    r2 = subprocess.run(["python", "-m", modname], capture_output=True, text=True)
    return path, r2.returncode, "module", r2.stdout, r2.stderr


def copy_figures():
    """Copy all Figures/ subfolders from each DIR into DEST_ROOT."""
    DEST_ROOT.mkdir(exist_ok=True)
    copied = 0
    for d in DIRS:
        src = Path(d) / "Figures"
        if not src.exists():
            continue
        dst = DEST_ROOT
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += 1
        print(f" Copied figures from {src} → {dst}")
    if copied == 0:
        print("  No 'Figures' folders found to copy.")
    else:
        print(f" Copied {copied} figure folder(s) into {DEST_ROOT}/")


if __name__ == "__main__":
    scripts = discover_scripts()
    if not scripts:
        print("No scripts found.")
        raise SystemExit(0)

    print(f"Running {len(scripts)} scripts using {os.cpu_count()} cores...\n")

    fails = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {
            ex.submit(run_script, s): s for s in scripts if "__init__" not in s.name
        }
        for fut in as_completed(futures):
            path, code, mode, out, err = fut.result()
            if code == 0:
                print(f" {path} [{mode}]")
            else:
                fails += 1
                print(f"\n {path} [{mode}] (exit {code})")
                payload = (err or out or "").strip()
                if payload:
                    print(payload[-MAX_CHARS:])
                else:
                    print("(no output captured)")

    if fails == 0:
        print("\n All scripts completed successfully.")
    else:
        print(f"\n {fails} script(s) failed.")

    # --- Post-run copy step ---
    copy_figures()
