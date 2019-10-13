import argparse

from pathlib import Path
from PeonyPackage.PeonyDb import MongoDb

# Imports for datasets upload
from HuffPost_news_dataset import (
    COLLECTION_NAME as HuffPost_collection_name,
    transorm_data as HuffPost_transformer,
    load_data as HuffPost_loader,
)
from Newsgroups_dataset import (
    COLLECTION_NAME as NewsGroups_collection_name,
    transorm_data as NewsGroups_transformer,
    load_data as NewsGroups_loader,
)
from Tweets_emotions_dataset import (
    COLLECTION_NAME as Tweets_collection_name,
    transorm_data as Tweets_transformer,
    load_data as Tweets_loader,
)
from Comments_dataset import (
    COLLECTION_NAME as Comments_collection_name,
    transorm_data as Comments_transformer,
    load_data as Comments_loader,
)
from Emotions_dataset import (
    COLLECTION_NAME as Emotions_collection_name,
    transorm_data as Emotions_transformer,
    load_data as Emotions_loader,
)

# args for different datasets
def input_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huffpost", help="Path to HuffPost dataset")
    parser.add_argument("--newsgroups", help="Path to 4 newsgroups dataset")
    parser.add_argument("--tweets", help="Path to 1600k tweets (emotional semantics)")
    parser.add_argument("--comments", help="Path to comments dataset")
    parser.add_argument("--emotions", help="Path to emotional texts dataset")
    return parser


# upload to mongo
def main():
    args = input_args().parse_args()

    api = MongoDb()

    api.load_data_to_database(
        HuffPost_collection_name,
        Path(args.huffpost),
        HuffPost_loader,
        HuffPost_transformer,
    )

    api.load_data_to_database(
        NewsGroups_collection_name,
        Path(args.newsgroups),
        NewsGroups_loader,
        NewsGroups_transformer,
    )

    api.load_data_to_database(
        Tweets_collection_name, Path(args.tweets), Tweets_loader, Tweets_transformer
    )

    api.load_data_to_database(
        Comments_collection_name,
        Path(args.comments),
        Comments_loader,
        Comments_transformer,
    )

    api.load_data_to_database(
        Emotions_collection_name,
        Path(args.emotions),
        Emotions_loader,
        Emotions_transformer,
    )


if __name__ == "__main__":
    main()
