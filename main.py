import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Hardcoded directories ---
DIRS = ["assignment_2"]

MAX_CHARS = 4000  # cap error output to avoid spam


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
    """Return (path, code, mode, stdout, stderr). mode in {'direct','module'}."""
    # 1) try direct
    r = subprocess.run(["python", str(path)], capture_output=True, text=True)
    if r.returncode == 0:
        return path, 0, "direct", r.stdout, r.stderr
    # 2) fallback: as module
    modname = to_module_name(path)
    r2 = subprocess.run(["python", "-m", modname], capture_output=True, text=True)
    return path, r2.returncode, "module", r2.stdout, r2.stderr


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
                    # print at most MAX_CHARS to keep output readable
                    print(payload[-MAX_CHARS:])
                else:
                    print("(no output captured)")

    if fails == 0:
        print("\n All scripts completed successfully.")
    else:
        print(f"\n {fails} script(s) failed.")
