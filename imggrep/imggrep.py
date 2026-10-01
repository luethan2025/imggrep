import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .argparse_utils import resolve_input
from .clip_utils import CLIPEmbedder
from .hash_utils import pHashEmbedder
from .os_utils import list_files
from .siglip_utils import SigLIPEmbedder

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple command-line interface."
    )
    parser.add_argument(
        "input",
        type=resolve_input,
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
        choices=["phash", "clip", "siglip"],
        help="Embedding method. Must be one of: `phash`, `clip`, `siglip`",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Maximum distance between two images.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not (target.exists() and target.is_dir()):
        raise ValueError("`--target` value is not a directory")

    return args

def main():
    args = parse_args()

    input = args.input

    match args.method:
        case "phash":
            embedder = pHashEmbedder()
        case "clip":
            embedder = CLIPEmbedder()
        case "siglip":
            embedder = SigLIPEmbedder()

    if isinstance(input, Path):
        embedder.set_reference_embeddings(Image.open(input))
    elif isinstance(input, str):
        embedder.set_reference_embeddings(input)
    else:
        raise ValueError("--input value is neither a Path or string")

    paths = list_files(args.target)

    matches = []
    for target in tqdm(
        paths,
        desc="Searching",
        bar_format="{desc}: |{bar}| {percentage:3.0f}% [{elapsed}]",
    ):
        if isinstance(input, str) or input.resolve() != target.resolve():
            try:
                is_similar = embedder.is_similar_to_reference_embeddings(
                    Image.open(target),
                    **(
                        {"distance": args.distance}
                        if args.distance is not None
                        else {}
                    ),
                )
                if is_similar:
                    matches.append(
                        target.resolve().relative_to(
                            Path.cwd(),
                            walk_up=True,
                        )
                    )
            except Exception as e:
                pass

if __name__ == "__main__":
    main()
