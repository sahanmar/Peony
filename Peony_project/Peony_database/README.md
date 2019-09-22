# Peony Database Architecture

## Database Description 

![MongoDatabase](https://github.com/sahanmar/Peony/blob/supporting_images/images/architecture_images/mongodb.png)

In this project we decided to work with NoSQL database. Our choice was MongoDb. The reason why we have chosen MongoDb is because of its simplicity and possibility of maintaining through Docker 

## Docker + MongoDb

Since Doker and MongoDb is perfect combination, Peony Database can be deployed with two lines of code (Remeber to activate `peony_project` environment)
1. Run Docker + MongoDb with `docker-compose up -d --build`
2. Load the data with `python3 fill_in_the_database.py --huffpost <path> --newsgroups <path> --tweets <path> --comments <path> --emotiones <path> --ner <path>`. Each parameter representes one dataset.
200k texts from Huffpost, 20 newsgroups datatset, 1600k tweets, emotions classification, NER_CONLL 2003

//TODO Add references to each dataset



## MongoDb Data Format 

MongoDb represents the data in BSON format behind the scenes but we will send and get JSON format data.

### Datasets and Their Instances

Here is JSON schema of how the data are stored and what a user will get as an output from a database. ([Understanding of JSON schema can be found here](https://json-schema.org/understanding-json-schema/))

```
{
	"title": "PeonyDatabase",
	"type": "object",
	"properties": {
		"datasetName": {
			"type": "string",
			"description": "Name of the dataset"
		},
		"datasetId": {
			"type": "string",
			"description": "Unique hash id that will be created automatically"  
		},
		"record": {
			"type": "object",
			"description": "All information about an instance",
			"properties": {
				"id": {
					"type": "string",
					"description": "Unique hash id that will be created automatically" 
				},
				"snippet": {
							"type": "string",
							"description": "Snippet of a text. Can be empty" 
				},
				"text": {
					"type": "object",
					"description": "Text instance that is used for a model",
					"properties" : {
						"title": {
							"type": "string",
							"description": "Title of a text. Can be empty"
						},
						"body": {
							"type": "string",
							"description": "Body of a text"
						},
					},
				},
				"label": {
					"type": "string",
					"description": "Label for an instance. Can be empty if this is not validation data"
				},
				"metadata": {
					"type": "object",
					"description": "Any additional metadata. Can be empty field"
				},
			},
		},
	},
}
```


### Models

In order to use specific model from PeonyBox in future, a user can store this model model in Peony Database. 

//TODO Figure out how models will be stored in the database 
