import argparse

from pathlib import Path
from HuffPost_news_dataset import load_data_to_database as HuffPost_loader
from Newsgroups_dataset import load_data_to_database as Newsgroups_loader


def input_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huffpost", help="Path to HuffPost dataset")
    parser.add_argument("--newsgroups", help="Path to 20 newsgroups dataset")
    return parser.parse_args()


def main():
    args = input_args()

    HuffPost_loader(Path(args.huffpost))
    Newsgroups_loader(Path(args.newsgroups))


if __name__ == "__main__":
    main()
