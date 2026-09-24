import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Simple command-line interface.")
    parser.add_argument(
        "--input",
        type="str",
        require=True,
        help="Path to image.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=10.,
        help="Maximum distance between two images.",
    )
    args = parser.parse_args()

    input = Path(args.input)
    if not(input.exists() and input.is_file()):
        raise ValueError("`--input` value is not a file")

    if not args.input.endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("`--input` value is not an image")

    if args.distance <= 0:
        raise ValueError("`--distance` value must be greater than 0")

    return args

def main():
    args = parse_args()

if __name__ == "__main__":
    main()
