"""Assemble the documentation site's sources for MkDocs (``mkdocs.yml``).

The site reuses the repository's Markdown as is: the README becomes the home page,
``docs/`` keeps its layout, and each demo folder brings its pages, GIF and charts.
Relative links to files that are not on the site (code, tests, configs) are rewritten
to GitHub, so they work from the published pages too.

    pip install -e ".[docs]"
    python scripts/build_site.py && mkdocs serve
"""

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "https://github.com/leduardoaraujo/NextAction"
RENAMED = {"README.md": "index.md", "README.pt-BR.md": "pt-BR.md"}
MEDIA = {".gif", ".png", ".jpg", ".jpeg", ".svg"}
DEMOS = ("pacman", "connect4", "doom", "orchestrator")
LINK = re.compile(r"(\]\()([^)\s]+)(\))")  # every link target, also inside badges


def site_files() -> dict[Path, Path]:
    """Map each repository file that goes on the site to its path in the site."""
    files = {ROOT / source: Path(target) for source, target in RENAMED.items()}
    for path in (ROOT / "docs").rglob("*.md"):
        files[path] = path.relative_to(ROOT)
    for demo in DEMOS:
        folder = ROOT / "demos" / demo
        for path in folder.rglob("*"):
            if path.suffix == ".md" or path.suffix in MEDIA:
                files[path] = path.relative_to(ROOT)
    return files


def rewrite(text: str, source: Path, target: Path, files: dict[Path, Path]) -> str:
    def replace(match: re.Match) -> str:
        prefix, link, suffix = match.groups()
        if re.match(r"^[a-z]+:|^#|^/", link):
            return match.group(0)
        path, _, anchor = link.partition("#")
        resolved = (source.parent / path).resolve()
        if resolved in files:
            new = Path(os.path.relpath(files[resolved], target.parent)).as_posix()
            return f"{prefix}{new}{'#' + anchor if anchor else ''}{suffix}"
        try:
            repository_path = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        return f"{prefix}{REPOSITORY}/{kind}/main/{repository_path}{'#' + anchor if anchor else ''}{suffix}"
    return LINK.sub(replace, text)


def build(output: Path) -> int:
    if output.exists():
        shutil.rmtree(output)
    files = site_files()
    for source, target in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md":
            text = source.read_text(encoding="utf-8")
            destination.write_text(rewrite(text, source, target, files), encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(source, destination)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "site_src")
    args = parser.parse_args()
    print(f"{build(args.output)} files written to {args.output}")


if __name__ == "__main__":
    main()
