import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple command-line interface.")
    parser.add_argument(
        "--input",
        type="str",
        require=True,
        help="Path to image.",
    )
    args = parser.parse_args()

    if not args.input.endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("`--input` value is not an image")

    return args

def main():
    args = parse_args()

if __name__ == "__main__":
    main()
