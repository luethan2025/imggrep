from pathlib import Path


def resolve_input(value: str) -> Path | str:
    path = Path(value)
    if path.is_file():
        return path
    return value


def list_files(
    dir: str | Path,
    ext: list[str] | None = None,
) -> list[Path]:
    if ext is None:
        ext = [".png", ".jpg", ".jpeg"]
    files = [f for f in Path(dir).rglob("*") if f.is_file() and f.suffix.lower() in ext]
    return files
