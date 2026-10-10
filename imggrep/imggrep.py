import sys

from .cli.config import run_config
from .cli.search import run_search


def main() -> None:
    if sys.argv[1:2] == ["config"]:
        run_config()
    else:
        run_search()


if __name__ == "__main__":
    main()
