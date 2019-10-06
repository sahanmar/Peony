import argparse

from pathlib import Path
from PeonyPackage.PeonyDb import MongoDb

# Imports for datasets upload
from embeddings.fasttext_embeddings import (
    COLLECTION_NAME as fasttext_collection_name,
    transorm_data as fasttext_transformer,
    load_data as fasttext_loader,
)

# args for different datasets
def input_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext", help="Path to fasttext pretarained model")
    return parser


# upload to mongo
def main():
    args = input_args().parse_args()

    api = MongoDb()

    api.load_data_to_database(
        fasttext_collection_name,
        Path(args.fasttext),
        fasttext_loader,
        fasttext_transformer,
    )


if __name__ == "__main__":
    main()
