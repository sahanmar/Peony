import argparse

from PeonyPackage.PeonyDb import MongoDb
from bson.objectid import ObjectId


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    return parser


def main():
    args  = input_args().parse_args()

    api = MongoDb()
    api.database["models_results"].delete_one({'_id': ObjectId(args.id)})


if __name__=="__main__":
    main()

