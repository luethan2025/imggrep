from pathlib import Path

def list_files(dir: str, ext: list[str]=[".png", ".jpg", ".jpeg"]) -> list[str]:
    files = [
        f for f in Path(dir).rglob("*")
            if f.is_file() and f.suffix.lower() in ext
    ]
    return files
