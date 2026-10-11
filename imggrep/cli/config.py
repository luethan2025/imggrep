import argparse
import json
import os
import sys
from pathlib import Path

from .default_configs import DEFAULT_PHASH_CONFIG

def config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "imggrep" / "config.json"
    return Path.home() / ".config" / "imggrep" / "config.json"


def load_config() -> dict[str, str | float | None]:
    path = config_path()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return write_default_config()
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"Configuration in {path} must be a JSON object")

    model_id = config.get("model_id")
    if model_id is not None and (
        not isinstance(model_id, str)
        or model_id not in {"clip", "google/siglip-base-patch16-224"}
    ):
        raise ValueError(f"Invalid model_id in {path}: {model_id!r}")

    distance = config.get("distance")
    if "distance" in config and (
        distance is None
        or isinstance(distance, bool)
        or not isinstance(distance, (int, float))
    ):
        raise ValueError(f"Invalid distance in {path}: expected a number")

    return config


def write_default_config() -> dict[str, str | float | None]:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DEFAULT_PHASH_CONFIG, indent=2) + "\n",
        encoding="utf-8",
    )
    return DEFAULT_PHASH_CONFIG.copy()


def parse_config_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="imggrep config")
    parser.add_argument(
        "parameter",
        nargs="?",
        choices=["default"],
        help="Write the default configuration.",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Select the default configuration.",
    )
    args = parser.parse_args(sys.argv[2:])

    return args


def run_config() -> None:
    args = parse_config_args()
    if args.parameter == "default" or args.default:
        write_default_config()
        print(f"imggrep configuration saved at {config_path()}")
