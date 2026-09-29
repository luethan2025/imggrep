import pathlib

def resolve_input(value: str) -> pathlib.Path | str:
    path = pathlib.Path(value)
    if path.is_file():
        return path
    return value
