import argparse
import json
import os
import sys
from pathlib import Path

from .default_configs import (
    DEFAULT_CLIP_CONFIG,
    DEFAULT_PHASH_CONFIG,
    DEFAULT_SIGLIP_CONFIG,
)


def config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "imggrep" / "config.json"
    return Path.home() / ".config" / "imggrep" / "config.json"


def load_config() -> dict[str, str | float | int | None]:
    path = config_path()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return write_default_config()
    except json.JSONDecodeError as error:
        raise ValueError("Unable to decode configuration file.") from error

    if not isinstance(config, dict):
        raise TypeError(
            f"Unable to decode configuration file. Expected a JSON object, received {type(config).__name__}"
        )

    model_id = config.get("model_id")
    if model_id is not None and (
        not isinstance(model_id, str)
        or not model_id.startswith(("openai/clip-", "google/siglip-"))
    ):
        raise ValueError(
            f"Unsupported model id: {model_id!r}. Must be a OpenAI CLIP model or Google SigLIP model."
        )

    distance = config.get("distance")
    if "distance" in config and (
        distance is None
        or isinstance(distance, bool)
        or not isinstance(distance, (int, float))
    ):
        raise ValueError(
            f"Unable to interpret distance. Expected a number, received {type(distance).__name__}."
        )

    batch_size = config.get("batch_size")
    if "batch_size" in config and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError(
            f"Unable to interpret batch_size. Expected a positive integer, received {type(batch_size).__name__}."
        )

    num_threads = config.get("num_threads")
    if "num_threads" in config and (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or num_threads < 1
    ):
        raise ValueError(
            f"Unable to interpret num_threads. Expected a positive integer, received {type(num_threads).__name__}."
        )

    dtype = config.get("dtype")
    if "dtype" in config and (
        not isinstance(dtype, str) or dtype not in ("no", "fp16", "bf16")
    ):
        raise ValueError(
            f"Unable to interpret dtype. Expected one of the following: 'no', 'fp16', 'bf16', received {type(num_threads).__name__}."
        )

    return config


def write_config(
    config: dict[str, str | float | int | None],
) -> dict[str, str | float | int | None]:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config.copy()


def write_default_config() -> dict[str, str | float | int | None]:
    return write_config(DEFAULT_PHASH_CONFIG)


def parse_config_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="imggrep config")
    parser.add_argument(
        "parameter",
        nargs="?",
        choices=["default"],
        help="Write a default configuration preset.",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        choices=["phash", "clip", "siglip"],
        help="Configuration preset to write (defaults to phash).",
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
    if args.parameter is not None or args.default:
        preset = args.preset or "phash"
        configs = {
            "phash": DEFAULT_PHASH_CONFIG,
            "clip": DEFAULT_CLIP_CONFIG,
            "siglip": DEFAULT_SIGLIP_CONFIG,
        }
        config = configs[preset]

    else:
        config = {}

        print("Configure model settings:")
        print("Please select a model:")
        print("  1. None")
        print("  2. openai/clip-vit-base-patch32")
        print("  3. google/siglip-base-patch16-224")

        choice = input("Your choice: ").strip() or "1"

        if choice == "1":
            model_id = None
        elif choice == "2":
            model_id = "openai/clip-vit-base-patch32"
        elif choice == "3":
            model_id = "google/siglip-base-patch16-224"
        else:
            raise ValueError("Invalid model selection")

        print(f"Using model_id: {model_id}")
        config["model_id"] = model_id

        print()

        print("Configure distance settings:")
        default_distance = 1 if model_id is None else 0.1

        choice = input("Distance: ").strip()

        if choice:
            distance = float(choice)
            if distance.is_integer():
                distance = int(distance)
        else:
            distance = default_distance

        print(f"Using distance: {distance}")
        config["distance"] = distance

        print()

        if model_id is None:
            default_num_threads = 4

            choice = input("Number of threads: ").strip()

            num_threads = int(choice) if choice else default_num_threads
            if num_threads < 1:
                raise ValueError("Number of threads must be a positive integer.")

            print(f"Using number of threads: {num_threads}")
            config["num_threads"] = num_threads
        else:
            print("Configure batch size settings:")

            default_batch_size = 16

            choice = input("Batch size: ").strip()

            batch_size = int(choice) if choice else default_batch_size
            if batch_size < 1:
                raise ValueError("Batch size must be a positive integer.")

            print(f"Using batch size: {batch_size}")
            config["batch_size"] = batch_size

            print()

            print("Configure dtype settings:")

            default_dtype = "fp16"

            choice = input("dtype: ").strip()

            dtype = choice if choice else default_dtype
            if dtype not in ("no", "fp16", "bf16"):
                raise ValueError("dtype must be one of: 'no', 'fp16', 'bf16'.")

            print(f"Using dtype: {dtype}")
            config["dtype"] = dtype

    write_config(config)
    print(f"imggrep configuration saved at {config_path()}")
