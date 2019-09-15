# Peony Database Architecture

## Database Description 

![MongoDatabase](https://github.com/sahanmar/Peony/blob/supporting_images/images/architecture_images/mongodb.png)

In this project we are decided to work with NoSql database. Our choice was MongoDb. The reason why we have chosen MongoDb is used because of its simplicity and possibility of working through Docker 

## Docker + MongoDb

//TODO Add description of working with Mongo through Docker and more images.


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
    		"properties": {
    			"id": {
    				"type": "string",
    				"description": "Unique hash id that will be created automatically" 
    			},
    			"text": {
    				"type": "string",
    				"description": "Text instance that is used for model"
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
