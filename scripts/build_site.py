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
import posixpath
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "https://github.com/leduardoaraujo/NextAction"
RENAMED = {"README.md": "index.md", "README.pt-BR.md": "index.pt.md"}
MEDIA = {".gif", ".png", ".jpg", ".jpeg", ".svg"}
DEMOS = ("pacman", "connect4", "doom", "orchestrator")
LINK = re.compile(r"(\]\()([^)\s]+)(\))")  # every link target, also inside badges


def site_files() -> dict[Path, Path]:
    """Map each repository file that goes on the site to its path in the site."""
    files = {ROOT / source: Path(target) for source, target in RENAMED.items()}
    for path in (ROOT / "docs").rglob("*"):
        if path.suffix == ".md" or path.suffix in MEDIA:
            files[path] = site_path(path)
    for demo in DEMOS:
        folder = ROOT / "demos" / demo
        for path in folder.rglob("*"):
            if path.suffix == ".md" or path.suffix in MEDIA:
                files[path] = site_path(path)
    return files


def site_path(path: Path) -> Path:
    """``name.pt-BR.md`` in the repository is ``name.pt.md`` for mkdocs-static-i18n."""
    relative = path.relative_to(ROOT)
    return relative.with_name(relative.name.replace(".pt-BR.md", ".pt.md"))


def is_portuguese(target: Path) -> bool:
    return target.name.endswith(".pt.md")


def page_url(target: Path) -> str:
    """The page's directory URL relative to the site root, e.g. ``pt/docs/api/``."""
    portuguese = is_portuguese(target)
    name = target.name.removesuffix(".pt.md").removesuffix(".md")
    folder = target.parent if name in ("index", "README") else target.parent / name
    url = (Path("pt") / folder) if portuguese else folder
    return "" if url == Path(".") else url.as_posix() + "/"


def rewrite(text: str, source: Path, target: Path, files: dict[Path, Path]) -> str:
    def replace(match: re.Match) -> str:
        prefix, link, suffix = match.groups()
        if re.match(r"^[a-z]+:|^#|^/", link):
            return match.group(0)
        path, _, anchor = link.partition("#")
        resolved = (source.parent / path).resolve()
        anchor = f"#{anchor}" if anchor else ""
        if resolved in files and resolved.suffix == ".md" and is_portuguese(files[resolved]) != is_portuguese(target):
            # Links between languages point to the other language's page itself.
            new = posixpath.relpath("/" + page_url(files[resolved]), "/" + page_url(target))
            return f"{prefix}{new}/{anchor}{suffix}"
        if resolved in files:
            new = Path(os.path.relpath(files[resolved], target.parent)).as_posix()
            return f"{prefix}{new}{anchor}{suffix}"
        try:
            repository_path = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        return f"{prefix}{REPOSITORY}/{kind}/main/{repository_path}{anchor}{suffix}"
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
