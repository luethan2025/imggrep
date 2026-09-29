import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .hash_utils import pHashEmbedder
from .os_utils import list_files

def parse_args():
    parser = argparse.ArgumentParser(description="Simple command-line interface.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to image.",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to target directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="phash",
        choices=["phash"],
        help="Embedding method. Must be one of: `phash`",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=10.,
        help="Maximum distance between two images.",
    )
    args = parser.parse_args()

    input = Path(args.input)
    if not (input.exists() and input.is_file()):
        raise ValueError("`--input` value is not a file")

    target = Path(args.target)
    if not (target.exists() and target.is_dir()):
        raise ValueError("`--target` value is not a directory")

    if args.input.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("`--input` value is not an image")

    if args.distance <= 0:
        raise ValueError("`--distance` value must be greater than 0")

    return args

def main():
    args = parse_args()

    max_distance = args.distance
    input = args.input

    match args.method:
        case "phash":
            embedder = pHashEmbedder()

    embedder.set_reference_embeddings(Image.open(input))
    paths = list_files(args.target)

    matches = []
    for target in tqdm(
        paths,
        desc="Searching",
        bar_format="{desc}: |{bar}| {percentage:3.0f}% [{elapsed}]"
    ):
        if input.resolve() != target.resolve():
            try:
                target_image_embeddings = embedder.embed(Image.open(target))
                is_similiar = embedder.is_similiar_to_reference_embeddings(
                    target_image_embeddings,
                    distance=max_distance
                )
                if is_similiar:
                    matches.append(target.resolve().relative_to(Path.cwd(), walk_up=True))
            except Exception as e:
                pass

if __name__ == "__main__":
    main()
