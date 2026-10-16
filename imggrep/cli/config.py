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

CONFIG_PRESETS = {
    "phash": DEFAULT_PHASH_CONFIG,
    "clip": DEFAULT_CLIP_CONFIG,
    "siglip": DEFAULT_SIGLIP_CONFIG,
}

SUPPORTED_MODELS = {
    "1": None,
    "2": "openai/clip-vit-base-patch32",
    "3": "google/siglip-base-patch16-224",
}

SUPPORTED_DTYPES = ("no", "fp16", "bf16")

CONFIG_PARAMETERS = (
    "model_id",
    "distance",
    "batch_size",
    "num_threads",
    "top_k",
    "dtype",
)

CONFIG_DEFAULTS = {
    "top_k": 10,
    "num_threads": 4,
    "batch_size": 16,
    "dtype": "fp16",
}


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
            "Unable to decode configuration file. "
            f"Expected a JSON object, received {type(config).__name__}."
        )

    model_id = config.get("model_id")

    if model_id is not None:
        if not isinstance(model_id, str):
            raise TypeError(
                "Unable to interpret model_id. Expected a string or None, "
                f"received {type(model_id).__name__}."
            )

        if not model_id.startswith(("openai/clip-", "google/siglip-")):
            raise ValueError(
                f"Unsupported model id: {model_id!r}. "
                "Must be an OpenAI CLIP model or Google SigLIP model."
            )

    distance = config.get("distance")

    if "distance" in config and (
        distance is None
        or isinstance(distance, bool)
        or not isinstance(distance, (int, float))
    ):
        raise TypeError(
            "Unable to interpret distance. Expected a number, "
            f"received {type(distance).__name__}."
        )

    for parameter in ("batch_size", "num_threads", "top_k"):
        value = config.get(parameter)

        if parameter not in config:
            continue

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Unable to interpret {parameter}. Expected an integer, "
                f"received {type(value).__name__}."
            )

        if value < 1:
            raise ValueError(
                f"Unable to interpret {parameter}. Expected a positive integer."
            )

    dtype = config.get("dtype")

    if "dtype" in config:
        if not isinstance(dtype, str):
            raise TypeError(
                "Unable to interpret dtype. Expected a string, "
                f"received {type(dtype).__name__}."
            )

        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                "Unable to interpret dtype. Expected one of "
                f"{SUPPORTED_DTYPES}, received {dtype!r}."
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


def write_default_config(
    preset: str = "phash",
) -> dict[str, str | float | int | None]:
    try:
        config = CONFIG_PRESETS[preset]
    except KeyError:
        raise ValueError(f"Unknown configuration preset: {preset}") from None

    return write_config(dict(config))


def prompt_integer(
    prompt: str,
    current: int,
    parameter: str,
) -> int:
    choice = input(f"{prompt}: ").strip()

    try:
        value = int(choice) if choice else current
    except ValueError as error:
        raise ValueError(f"{parameter} must be an integer.") from error

    if value < 1:
        raise ValueError(f"{parameter} must be a positive integer.")

    return value


def prompt_distance(current: float) -> int | float:
    choice = input("Distance: ").strip()

    try:
        value = float(choice) if choice else current
    except ValueError as error:
        raise ValueError("distance must be a number.") from error

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def prompt_model_id(current_model: str | None) -> str | None:
    print("Please select a model:")
    print("  1. None")
    print("  2. openai/clip-vit-base-patch32")
    print("  3. google/siglip-base-patch16-224")

    current_choice = next(
        (key for key, model in SUPPORTED_MODELS.items() if model == current_model),
        "1",
    )

    choice = input("Your choice: ").strip()
    choice = choice or current_choice

    if choice not in SUPPORTED_MODELS:
        raise ValueError("Invalid model selection.")

    return SUPPORTED_MODELS[choice]


def prompt_dtype(current: str = "fp16") -> str:
    choice = input("torch.dtype: ").strip()
    value = choice or current

    if value not in SUPPORTED_DTYPES:
        raise ValueError("dtype must be one of: 'no', 'fp16', 'bf16'.")

    return value


def configure_parameter(parameter: str) -> None:
    config = load_config()

    if parameter == "model_id":
        model_id = prompt_model_id(config.get("model_id"))
        config["model_id"] = model_id

        if model_id is None:
            config.pop("batch_size", None)
            config.pop("dtype", None)
            config.setdefault("num_threads", 4)
            config.setdefault("distance", 1)
        else:
            config.pop("num_threads", None)
            config.setdefault("batch_size", 16)
            config.setdefault("dtype", "fp16")
            config.setdefault("distance", 0.1)

    elif parameter == "top_k":
        config["top_k"] = prompt_integer(
            "Top-K",
            config.get("top_k", 10),
            "top_k",
        )

    elif parameter == "distance":
        default = 1 if config.get("model_id") is None else 0.1
        config["distance"] = prompt_distance(config.get("distance", default))

    elif parameter == "num_threads":
        config["num_threads"] = prompt_integer(
            "Number of threads",
            config.get("num_threads", 4),
            "num_threads",
        )

    elif parameter == "batch_size":
        config["batch_size"] = prompt_integer(
            "Batch size",
            config.get("batch_size", 16),
            "batch_size",
        )

    elif parameter == "dtype":
        config["dtype"] = prompt_dtype(config.get("dtype", "fp16"))

    else:
        raise ValueError(f"Unsupported configuration parameter: {parameter}")

    write_config(config)

    print(f"Using {parameter}: {config.get(parameter)}")
    print(f"imggrep configuration saved at {config_path()}")


def configure_interactive() -> None:
    config = load_config()

    print("Configure model settings:")

    model_id = prompt_model_id(config.get("model_id"))
    config["model_id"] = model_id

    print(f"Using model_id: {model_id}")
    print()

    config["top_k"] = prompt_integer(
        "Top-K",
        config.get("top_k", 10),
        "top_k",
    )
    print(f"Using Top-K: {config['top_k']}")
    print()

    default_distance = 1 if model_id is None else 0.1

    config["distance"] = prompt_distance(config.get("distance", default_distance))
    print(f"Using distance: {config['distance']}")
    print()

    if model_id is None:
        config.pop("batch_size", None)
        config.pop("dtype", None)

        config["num_threads"] = prompt_integer(
            "Number of threads",
            config.get("num_threads", 4),
            "num_threads",
        )
        print(f"Using number of threads: {config['num_threads']}")

    else:
        config.pop("num_threads", None)

        config["batch_size"] = prompt_integer(
            "Batch size",
            config.get("batch_size", 16),
            "batch_size",
        )
        print(f"Using batch size: {config['batch_size']}")
        print()

        config["dtype"] = prompt_dtype(config.get("dtype", "fp16"))
        print(f"Using dtype: {config['dtype']}")

    write_config(config)
    print(f"imggrep configuration saved at {config_path()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="imggrep config")
    subparsers = parser.add_subparsers(dest="command")

    default_parser = subparsers.add_parser(
        "default",
        help="Write a default configuration preset.",
    )
    default_parser.add_argument(
        "preset",
        nargs="?",
        choices=tuple(CONFIG_PRESETS),
        default="phash",
        help="Configuration preset to write.",
    )

    set_parser = subparsers.add_parser(
        "set",
        help="Update a configuration setting.",
    )
    set_parser.add_argument(
        "parameter",
        choices=CONFIG_PARAMETERS,
        help="Configuration parameter to update.",
    )

    return parser.parse_args(sys.argv[2:])


def run_config() -> None:
    args = parse_args()

    if args.command is None:
        configure_interactive()

    elif args.command == "default":
        write_default_config(args.preset)
        print(f"imggrep configuration saved at {config_path()}")

    elif args.command == "set":
        configure_parameter(args.parameter)
