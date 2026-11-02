#!/usr/bin/env python3
"""Setup for the repo: install the packages, clone the pinned repos, download and unpack the data,
fetch the weights, and check that everything is where the code expects it.

Run it inside the conda environment (works the same on Windows, macOS and Linux):

    conda env create -f environment.yml
    conda activate aivis
    python scripts/aivis_setup.py all

Single steps:

    python scripts/aivis_setup.py install [--torch auto|cpu|cuda]
    python scripts/aivis_setup.py clone [--with-optional]
    python scripts/aivis_setup.py data [--skip-eval] [--keep-zips] [--dev-zip PATH] [--eval-zip PATH]
    python scripts/aivis_setup.py weights
    python scripts/aivis_setup.py check

The script itself only needs the standard library (Python 3.11+). Everything that gets downloaded
is pinned in config/external.toml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("This needs Python 3.11 or newer. Create the environment first: "
             "conda env create -f environment.yml && conda activate aivis")

ROOT = Path(__file__).resolve().parents[1]
CFG = tomllib.loads((ROOT / "config" / "external.toml").read_text(encoding="utf-8"))
LOCAL_CFG = ROOT / "config" / "local.toml"   # machine-specific settings, not in git
IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"
TORCH = ["torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0"]   # keep in sync with requirements.txt
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "cuda": "https://download.pytorch.org/whl/cu128"}


def local_settings() -> dict:
    return tomllib.loads(LOCAL_CFG.read_text(encoding="utf-8")) if LOCAL_CFG.exists() else {}


def data_root() -> Path:
    """Same rule as aivis/paths.py: AIVIS_DATA, else data_dir in config/local.toml, else data/."""
    return Path(os.environ.get("AIVIS_DATA") or local_settings().get("data_dir") or ROOT / "data")


def say(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


class SetupError(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    cmd = ["git", *args]
    if capture:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    subprocess.run(cmd, cwd=cwd, check=True)
    return ""


# -------------------------------------------------------------------------- packages

def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True, cwd=ROOT)


def torch_flavour(choice: str) -> str | None:
    """Which PyTorch index to use. None means plain PyPI."""
    if choice == "auto":
        if IS_WIN and shutil.which("nvidia-smi"):
            return "cuda"      # PyPI only has CPU builds for Windows
        return None            # Linux: PyPI build has CUDA; macOS: PyPI build has MPS
    if IS_MAC:
        if choice == "cuda":
            warn("there is no CUDA on macOS, installing the normal build")
        return None
    return choice


def cmd_install(args) -> None:
    if not os.environ.get("CONDA_PREFIX"):
        warn("no conda environment active. Run `conda activate aivis` first, "
             "unless you really want the packages in this Python.")
    if sys.version_info[:2] != (3, 11):
        warn(f"this is Python {sys.version.split()[0]}, everything was tested with 3.11 (environment.yml)")
    flavour = torch_flavour(args.torch)
    say("PyTorch 2.8" + (f" ({flavour} build)" if flavour else ""))
    pip("install", *TORCH, *(["--index-url", TORCH_INDEX[flavour]] if flavour else []))
    say("packages from requirements.txt")
    pip("install", "-r", "requirements.txt")
    wv = ROOT / CFG["repos"]["whalevad"]["path"]
    if not wv.exists():
        raise SetupError(f"{wv} is missing, run the clone step first")
    say("whalevad and aivis (editable)")
    pip("install", "--no-deps", "-e", str(wv), "-e", ".")
    say("Jupyter kernel \"Python (aivis)\"")
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "aivis",
                    "--display-name", "Python (aivis)"], check=False)


def cmd_deepvoice(args) -> None:
    """Second conda environment for DeepVoice (Python 3.10, their own requirements)."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise SetupError("conda not found, DeepVoice needs its own environment (aivis-deepvoice)")
    envs = subprocess.run([conda, "env", "list"], text=True, capture_output=True).stdout
    if not any(line.split()[:1] == ["aivis-deepvoice"] for line in envs.splitlines()):
        say("creating the conda environment aivis-deepvoice (Python 3.10)")
        subprocess.run([conda, "create", "-y", "-n", "aivis-deepvoice", "python=3.10", "pip"], check=True)
    flavour = torch_flavour(args.torch)
    run = [conda, "run", "--no-capture-output", "-n", "aivis-deepvoice", "python", "-m", "pip", "install"]
    subprocess.run([*run, *TORCH, *(["--index-url", TORCH_INDEX[flavour]] if flavour else [])], check=True)
    req = ROOT / CFG["repos"]["deepvoice"]["path"] / "requirements.txt"
    subprocess.run([*run, "-r", str(req)], check=True)


def set_data_dir(path: str) -> None:
    folder = Path(path).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    LOCAL_CFG.write_text("# machine-specific, not in git\n"
                         f"data_dir = {json.dumps(str(folder))}\n", encoding="utf-8")
    os.environ.pop("AIVIS_DATA", None)
    say(f"data folder: {folder} (saved in config/local.toml)")


def cmd_all(args) -> None:
    if not shutil.which("git"):
        raise SetupError("git not found. Install it (https://git-scm.com, or `conda install git`) and run again.")
    if args.data_dir:
        set_data_dir(args.data_dir)
    args.with_optional = args.with_deepvoice
    say("model repos (pinned in config/external.toml)")
    cmd_clone(args)
    if not args.no_install:
        cmd_install(args)
    if args.with_deepvoice:
        cmd_deepvoice(args)
    if not args.no_data:
        say(f"data into {data_root()} (checked against the md5 in config/external.toml)")
        cmd_data(args)
    say("weights")
    try:
        cmd_weights(args)
    except SetupError as e:
        warn(f"{e} Everything else still works.")
    # fresh process, so packages installed a moment ago are importable
    check = [sys.executable, str(Path(__file__).resolve()), "check"]
    check += ["--skip-data"] if args.no_data else (["--skip-eval"] if args.skip_eval else [])
    subprocess.run(check, cwd=ROOT)
    if not args.no_smoke_test:
        say("quick test on a synthetic recording (all three models, CPU, about a minute)")
        r = subprocess.run([sys.executable, "-m", "aivis.smoke_test"], cwd=ROOT)
        if r.returncode:
            warn("the quick test found problems, see above")
    print("\nDone. Use the environment with:  conda activate aivis")


# ----------------------------------------------------------------------------- repos

def git_head(path: Path) -> str | None:
    try:
        return git("-C", str(path), "rev-parse", "HEAD", capture=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty(path: Path) -> bool:
    try:
        return bool(git("-C", str(path), "status", "--porcelain", "--untracked-files=no", capture=True))
    except subprocess.CalledProcessError:
        return False


def commit_exists(path: Path, commit: str) -> bool:
    return subprocess.run(["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


def cmd_clone(args) -> None:
    for key, repo in CFG["repos"].items():
        if repo.get("optional") and not args.with_optional:
            continue
        path, commit = ROOT / repo["path"], repo["commit"]
        if path.is_symlink() or (path / ".git").exists():
            head = git_head(path)
            if head == commit:
                say(f"{key}: at {commit[:7]}")
                continue
            if path.is_symlink():
                warn(f"{key}: {path.name} links to your own checkout ({(head or '?')[:7]}), leaving it alone. "
                     f"Pinned commit is {commit[:7]}.")
                continue
            if git_dirty(path):
                warn(f"{key}: {path.name} has local changes, not touching it. "
                     f"Pinned {commit[:7]}, checked out {(head or '?')[:7]}.")
                continue
            say(f"{key}: moving to pinned commit {commit[:7]}")
            if not commit_exists(path, commit):
                git("-C", str(path), "fetch", "--quiet", "--all", "--tags")
            git("-C", str(path), "checkout", "--quiet", "--detach", commit)
        else:
            if path.exists() and any(path.iterdir()):
                raise SetupError(f"{path} exists but is not a git checkout. Move it away and run again.")
            say(f"{key}: cloning {repo['url']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--quiet", repo["url"], str(path))
            git("-C", str(path), "checkout", "--quiet", "--detach", commit)


# ------------------------------------------------------------------------- downloads

def zenodo_url(record: int, file: str) -> str:
    base = os.environ.get("AIVIS_ZENODO_BASE", "https://zenodo.org").rstrip("/")   # a mirror, if Zenodo is slow
    return f"{base}/records/{record}/files/{file}?download=1"


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, md5: str | None = None, sha256_prefix: str | None = None) -> bool:
    if not path.exists():
        return False
    if md5 and file_hash(path, "md5") != md5:
        return False
    if sha256_prefix and not file_hash(path, "sha256").startswith(sha256_prefix):
        return False
    return True


def check_space(folder: Path, need_gb: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(folder).free / 1e9
    if free_gb < need_gb:
        raise SetupError(f"Only {free_gb:.1f} GB free in {folder}, about {need_gb:.1f} GB needed. "
                         f"Free some space or put the data on a bigger disk with --data-dir PATH")


def _curl(url: str, part: Path, resume: bool) -> tuple[int, str]:
    """Returns curl's exit code and the last HTTP status."""
    cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5", "--connect-timeout", "30",
           "-A", "aivis-setup", "-o", str(part), "-w", "%{http_code}"]
    if resume:
        cmd += ["-C", "-"]
    cmd += ["--progress-bar"] if sys.stderr.isatty() else ["-sS"]
    cmd.append(url)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return r.returncode, (r.stdout or "").strip()[-3:]


def _urllib(url: str, part: Path) -> int:  # fallback when curl is missing
    import urllib.request
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "aivis-setup", "Range": f"bytes={start}-"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            mode = "ab" if r.status == 206 else "wb"
            with open(part, mode) as f:
                shutil.copyfileobj(r, f, length=1 << 22)
        return 0
    except Exception as e:  # noqa: BLE001
        warn(str(e))
        return 1


def fetch(url: str, dest: Path, md5: str | None = None, sha256_prefix: str | None = None,
          approx_gb: float | None = None, space_factor: float = 1.1) -> Path:
    """Download url to dest (resumable), verify the checksum, return dest."""
    if dest.exists():
        if verify(dest, md5, sha256_prefix):
            say(f"{dest.name}: already downloaded and verified")
            return dest
        warn(f"{dest.name}: checksum does not match, downloading again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if approx_gb:
        check_space(dest.parent, approx_gb * space_factor)
    part = dest.with_name(dest.name + ".part")
    say(f"downloading {dest.name}" + (f" (about {approx_gb} GB)" if approx_gb else ""))
    for attempt in range(1, 6):
        status = ""
        if shutil.which("curl"):
            code, status = _curl(url, part, resume=part.exists() and part.stat().st_size > 0)
            if code == 33:  # server refuses to resume, start over
                part.unlink(missing_ok=True)
                code, status = _curl(url, part, resume=False)
        else:
            code = _urllib(url, part)
        if code == 0 or verify(part, md5, sha256_prefix):
            break
        if status.startswith("4") and status not in ("408", "416", "429"):
            raise SetupError(f"{url} answered HTTP {status}. Check the link in config/external.toml, "
                             f"or whether your network blocks the site.")
        wait = 10 * attempt
        warn(f"download stopped (exit code {code}), trying again in {wait} s ({attempt}/5)")
        time.sleep(wait)
    else:
        raise SetupError(f"could not download {url}")
    if md5 or sha256_prefix:
        say(f"{dest.name}: checking the checksum")
        if not verify(part, md5, sha256_prefix):
            part.unlink(missing_ok=True)
            raise SetupError(f"{dest.name}: checksum does not match after the download. If this happens again, "
                             f"the file on Zenodo may have changed: compare with config/external.toml")
    part.replace(dest)
    return dest


# ------------------------------------------------------------------------------ data

def find_root(base: Path, marker: list[str], max_depth: int = 4) -> Path | None:
    """First folder below base (breadth first) that contains all marker folders."""
    level = [base]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            if all((d / m).is_dir() for m in marker):
                return d
            nxt += sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(("__MACOSX", ".")))
        level = nxt
    return None


def unpack(zip_path: Path, target: Path, marker: list[str]) -> None:
    tmp = target.parent / f".unpack_{target.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    say(f"unpacking {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist()
                   if not m.filename.startswith("__MACOSX/") and "/._" not in m.filename]
        for i, m in enumerate(members, 1):
            z.extract(m, tmp)
            if i % 500 == 0 or i == len(members):
                print(f"    {i}/{len(members)} files", flush=True)
    root = find_root(tmp, marker)
    if root is None:
        raise SetupError(f"{zip_path.name} does not contain the expected folders {marker}; look into {tmp}")
    root.rename(target)
    shutil.rmtree(tmp, ignore_errors=True)


def looks_complete(target: Path, marker: list[str]) -> bool:
    return all((target / m).is_dir() and any((target / m).iterdir()) for m in marker)


def cmd_data(args) -> None:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    zips = {"dev": args.dev_zip, "eval": args.eval_zip}
    for key in ("dev", "eval"):
        if key == "eval" and args.skip_eval:
            continue
        d = CFG["data"][key]
        target = root / d["target"]
        stamp = target / ".aivis_complete"
        if stamp.exists():
            say(f"{key}: already unpacked in {target}")
            continue
        if target.exists() and looks_complete(target, d["marker"]):
            say(f"{key}: found data in {target}, not downloading")
            try:
                stamp.write_text(json.dumps({"source": "found existing data", "date": now()}))
            except OSError:   # read-only shared folder, fine
                pass
            continue
        if target.exists():
            raise SetupError(f"{target} exists but looks incomplete. Move it away and run again.")
        own_zip = zips[key] is not None
        zip_path = Path(zips[key]) if own_zip else root / "downloads" / d["file"]
        if own_zip:
            say(f"{key}: using {zip_path}, checking the checksum")
            if not verify(zip_path, md5=d["md5"]):
                raise SetupError(f"{zip_path} does not match the checksum in config/external.toml")
        else:
            # zip and unpacked copy exist side by side for a moment
            fetch(zenodo_url(d["zenodo_record"], d["file"]), zip_path, md5=d["md5"],
                  approx_gb=d["approx_gb"], space_factor=2.2)
        unpack(zip_path, target, d["marker"])
        stamp.write_text(json.dumps({"source": d["file"], "md5": d["md5"], "date": now()}))
        if not own_zip and not args.keep_zips:
            zip_path.unlink()
            say(f"{key}: deleted {zip_path.name} (use --keep-zips to keep it)")


# --------------------------------------------------------------------------- weights

def cmd_weights(args) -> None:
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    failed = []
    for key, w in CFG["weights"].items():
        dest = wdir / w["file"]
        try:
            if "from_repo" in w:
                src = ROOT / CFG["repos"][w["from_repo"]]["path"] / w["repo_path"]
                if not src.exists():
                    raise SetupError(f"{src} is missing, run the clone step first")
                if not dest.exists() or file_hash(dest, "sha256") != file_hash(src, "sha256"):
                    shutil.copy2(src, dest)
                say(f"{key}: {dest.name} from the {w['from_repo']} repo")
            elif "zenodo_record" in w:
                fetch(zenodo_url(w["zenodo_record"], w["file"]), dest, md5=w.get("md5"))
            else:
                fetch(w["url"], dest, md5=w.get("md5"), sha256_prefix=w.get("sha256_prefix"))
        except (SetupError, OSError) as e:
            warn(f"{key}: {e}")
            failed.append(key)
    sums = [f"{file_hash(p, 'sha256')}  {p.name}" for p in sorted(wdir.glob("*.pt"))]
    (wdir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    if failed:
        raise SetupError(f"weights missing for: {', '.join(failed)}. Run the weights step again later "
                         f"or put the files into weights/ by hand (names in config/external.toml).")


# ----------------------------------------------------------------------------- check

def count_wavs(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.wav")) if folder.is_dir() else 0


def cmd_check(args) -> None:
    problems = []
    print("\nRepos")
    for key, repo in CFG["repos"].items():
        path = ROOT / repo["path"]
        head = git_head(path) if path.exists() else None
        if head is None:
            state = "not cloned" + (" (optional)" if repo.get("optional") else "")
            if not repo.get("optional"):
                problems.append(f"repo {key} missing")
        elif head == repo["commit"]:
            state = f"{head[:7]} (pinned)" + (", local changes" if git_dirty(path) else "")
        else:
            state = f"{head[:7]} (pinned is {repo['commit'][:7]})"
        print(f"  {key:10} {state}")

    print("\nData", f"({data_root()})")
    dev, ev = data_root() / "dev", data_root() / "eval"
    expected = CFG.get("expected_files", {})
    for split, base in (("train", dev / "train"), ("validation", dev / "validation"), ("evaluation", ev)):
        for site in CFG["sites"][split]:
            n = count_wavs(base / "audio" / site)
            ann = (base / "annotations" / f"{site}.csv").exists()
            note = ""
            if n == 0:
                note = "missing"
            elif site in expected and n != expected[site]:
                note = f"expected {expected[site]} files"
            if split != "evaluation" and n and not ann:
                note = (note + ", " if note else "") + "no annotations"
            print(f"  {split:10} {site:20} {n:5d} wav  {note}")
            if n == 0 and not args.skip_data and not (split == "evaluation" and args.skip_eval):
                problems.append(f"no audio for {site}")

    print("\nWeights")
    for key, w in CFG["weights"].items():
        p = ROOT / "weights" / w["file"]
        if p.exists():
            print(f"  {key:14} {w['file']}  sha256 {file_hash(p, 'sha256')[:12]}")
        else:
            print(f"  {key:14} missing")
            problems.append(f"weights {key} missing")

    print("\nPython packages")
    for mod in ("torch", "torchaudio", "ultralytics", "whalevad", "numpy", "pandas", "soundfile"):
        try:
            m = __import__(mod)
            extra = ""
            if mod == "torch":
                extra = f"  cuda={m.cuda.is_available()}"
                mps = getattr(m.backends, "mps", None)
                extra += f"  mps={bool(mps and mps.is_available())}"
            print(f"  {mod:12} {getattr(m, '__version__', '?')}{extra}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:12} not importable ({type(e).__name__})")
            problems.append(f"package {mod}")

    print()
    if problems:
        print("Not complete:", "; ".join(problems))
        if args.strict:
            sys.exit(1)
    else:
        print("Everything is in place.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_opts(x):
        x.add_argument("--skip-eval", action="store_true", help="skip only the evaluation set")
        x.add_argument("--keep-zips", action="store_true", help="keep the zips after unpacking")
        x.add_argument("--dev-zip", help="use an already downloaded development set zip")
        x.add_argument("--eval-zip", help="use an already downloaded evaluation set zip")

    def torch_opt(x):
        x.add_argument("--torch", choices=["auto", "cpu", "cuda"], default="auto",
                       help="PyTorch build. auto: CUDA on Linux and on Windows with an NVIDIA GPU, else CPU/MPS")

    a = sub.add_parser("all", help="everything: repos, packages, data, weights, check, quick test")
    a.add_argument("--data-dir", help="keep the data in this folder (bigger disk, shared folder); "
                                      "saved in config/local.toml")
    data_opts(a)
    torch_opt(a)
    a.add_argument("--no-data", action="store_true", help="skip the data (about 6.7 GB download)")
    a.add_argument("--no-install", action="store_true", help="skip the pip installs")
    a.add_argument("--with-deepvoice", action="store_true",
                   help="also DeepVoice, in a second conda environment aivis-deepvoice")
    a.add_argument("--no-smoke-test", action="store_true")
    i = sub.add_parser("install", help="pip install PyTorch, requirements.txt, whalevad and aivis")
    torch_opt(i)
    c = sub.add_parser("clone", help="clone the pinned repos into external/")
    c.add_argument("--with-optional", action="store_true", help="also clone optional repos (DeepVoice)")
    d = sub.add_parser("data", help="download and unpack the dev and eval sets")
    data_opts(d)
    sub.add_parser("weights", help="fetch the model weights into weights/")
    k = sub.add_parser("check", help="show what is in place")
    k.add_argument("--skip-eval", action="store_true")
    k.add_argument("--skip-data", action="store_true", help="do not count missing data as a problem")
    k.add_argument("--strict", action="store_true", help="exit with an error if something is missing")
    args = p.parse_args()
    cmds = {"all": cmd_all, "install": cmd_install, "clone": cmd_clone, "data": cmd_data,
            "weights": cmd_weights, "check": cmd_check}
    try:
        cmds[args.cmd](args)
    except SetupError as e:
        sys.exit(f"ERROR: {e}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: {' '.join(map(str, e.cmd))} failed (exit code {e.returncode})")


if __name__ == "__main__":
    main()
