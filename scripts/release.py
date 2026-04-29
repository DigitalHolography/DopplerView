import subprocess
import sys
import tomllib
import argparse
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
BUILD_SCRIPT = SCRIPT_DIR / "build_installer.py"
BUMPVERSION = SCRIPT_DIR / "bumpversion.cfg"


# -------------------------
# Utils
# -------------------------
def run(cmd, check=True):
    print(f"> {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=check)


def get_version():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


# -------------------------
# Safety
# -------------------------
def check_git_clean():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout)
        raise RuntimeError("Git working directory not clean")


def check_tag_exists(version):
    result = subprocess.run(
        ["git", "tag", "--list", f"{version}"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"Tag {version} already exists")


# -------------------------
# Version bump
# -------------------------
def bump_version(part):
    run(["bump2version", part, "--config-file", str(BUMPVERSION)])


# -------------------------
# Build
# -------------------------
def build_installer():
    run([sys.executable, str(BUILD_SCRIPT)])


# -------------------------
# Git
# -------------------------
def create_tag(version):
    run(["git", "tag", f"{version}"])
    run(["git", "push"])
    run(["git", "push", "origin", f"{version}"])


# -------------------------
# Changelog
# -------------------------
def generate_changelog(version):
    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "--no-decorate", "HEAD~10..HEAD"],
            text=True,
        )
    except Exception:
        log = "No changelog available"

    path = PROJECT_ROOT / "CHANGELOG.md"
    if path.exists():
        content = path.read_text()
    else:
        content = "# Changelog\n"

    content += f"\n## {version}\n\n{log}\n"
    path.write_text(content)

# -------------------------
# Reset
# -------------------------
def clean_build_artifacts():
    for folder in ["dist", "build"]:
        path = PROJECT_ROOT / folder
        if path.exists():
            print(f"Removing {path}")
            shutil.rmtree(path)

def reset_last_release():
    # Check last commit message
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        capture_output=True,
        text=True,
    )
    last_msg = result.stdout.strip()

    if "Bump version" not in last_msg:
        raise RuntimeError(
            "Last commit does not look like a bumpversion commit. Aborting reset."
        )

    print("Resetting last commit...")
    run(["git", "reset", "--hard", "HEAD~1"])

    # Clean build artifacts (optional but recommended)
    clean_build_artifacts()

    print("Reset complete.")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["patch", "minor", "major"], help="Part of the version to bump")
    parser.add_argument("--pre", action="store_true", help="Pre-release mode (no tag/push)")
    parser.add_argument("--finalize", action="store_true", help="Finalize pre-release (tag/push without bumping version)")
    parser.add_argument("--reset", action="store_true", help="Undo last pre-release")
    args = parser.parse_args()

    if args.reset:
        reset_last_release()
        return

    # 1. Safety
    check_git_clean()

    # 2. Bump version
    if not args.finalize:
        bump_version(args.part)

    # 3. Read version
    version = get_version()
    print(f"Preparing {version}")

    # 4. Safety again
    check_tag_exists(version)

    # 5. Build installer
    build_installer()

    if args.pre:
        print("Pre-release mode: skipping tag and push")
        print(f"Pre-release build for {version} complete")
        return

    # 6. Changelog
    generate_changelog(version)

    # 7. Tag + push
    create_tag(version)

    print(f"Release {version} complete")


if __name__ == "__main__":
    main()