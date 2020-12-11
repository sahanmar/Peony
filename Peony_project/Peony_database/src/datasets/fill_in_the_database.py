import argparse

from pathlib import Path
from PeonyPackage.PeonyDb import MongoDb

# Imports for datasets upload
from Peony_database.src.datasets.HuffPost_news_dataset import (
    COLLECTION_NAME as HuffPost_collection_name,
    transorm_data as HuffPost_transformer,
    load_data as HuffPost_loader,
)
from Peony_database.src.datasets.Newsgroups_dataset import (
    COLLECTION_NAME as NewsGroups_collection_name,
    transorm_data as NewsGroups_transformer,
    load_data as NewsGroups_loader,
)
from Peony_database.src.datasets.Tweets_emotions_dataset import (
    COLLECTION_NAME as Tweets_collection_name,
    transorm_data as Tweets_transformer,
    load_data as Tweets_loader,
)
from Peony_database.src.datasets.Comments_dataset import (
    COLLECTION_NAME as Comments_collection_name,
    transorm_data as Comments_transformer,
    load_data as Comments_loader,
)
from Peony_database.src.datasets.Emotions_dataset import (
    COLLECTION_NAME as Emotions_collection_name,
    transorm_data as Emotions_transformer,
    load_data as Emotions_loader,
)

from Peony_database.src.datasets.fake_news import (
    COLLECTION_NAME as fake_news_collection_name,
    transorm_data as fake_news_transformer,
    load_data as fake_news_loader,
)

from Peony_database.src.datasets.fake_news_detection import (
    COLLECTION_NAME as fake_news_detection_collection_name,
    transorm_data as fake_news_detection_transformer,
    load_data as fake_news_detection_loader,
)

# args for different datasets
def input_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huffpost", help="Path to HuffPost dataset")
    parser.add_argument("--newsgroups", help="Path to 4 newsgroups dataset")
    parser.add_argument("--tweets", help="Path to 1600k tweets (emotional semantics)")
    parser.add_argument("--comments", help="Path to comments dataset")
    parser.add_argument("--emotions", help="Path to emotional texts dataset")
    parser.add_argument("--fake_news", help="Path to fake news dataset")
    parser.add_argument("--fake_news_detection", help="Path to fake news detection dataset")
    return parser


# upload to mongo
def main():
    args = input_args().parse_args()

    api = MongoDb()

    if args.huffpost:
        api.load_data_to_database(
            HuffPost_collection_name,
            Path(args.huffpost),
            HuffPost_loader,
            HuffPost_transformer,
        )

    if args.newsgroups:
        api.load_data_to_database(
            NewsGroups_collection_name,
            Path(args.newsgroups),
            NewsGroups_loader,
            NewsGroups_transformer,
        )

    if args.tweets:
        api.load_data_to_database(
            Tweets_collection_name, Path(args.tweets), Tweets_loader, Tweets_transformer
        )

    if args.comments:
        api.load_data_to_database(
            Comments_collection_name,
            Path(args.comments),
            Comments_loader,
            Comments_transformer,
        )

    if args.emotions:
        api.load_data_to_database(
            Emotions_collection_name,
            Path(args.emotions),
            Emotions_loader,
            Emotions_transformer,
        )

    if args.fake_news:
        api.load_data_to_database(
            fake_news_collection_name,
            Path(args.fake_news),
            fake_news_loader,
            fake_news_transformer,
        )

    if args.fake_news_detection:
        api.load_data_to_database(
            fake_news_detection_collection_name,
            Path(args.fake_news_detection),
            fake_news_detection_loader,
            fake_news_detection_transformer,
        )


if __name__ == "__main__":
    main()
