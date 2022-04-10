from bson.json_util import dumps
from PeonyPackage.PeonyDb import MongoDb

if __name__ == '__main__':
    api = MongoDb()
    cursor = api.get_model_results({})
    with open('collection.json', 'w') as file:
        for document in cursor:
            file.write(dumps(document))
            file.write('\n')
