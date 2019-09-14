# Peony Project Architecture

## Database

![database|512x512,40%](https://github.com/sahanmar/Peony/blob/supporting_images/images/architecture_images/database.png)

In order to make everything consistent and let the models work with the same input format we decided to create a database that will store all the data in JSON format for the purposes of machine learning and visualizing components. 

## PeonyBox

![peonybox|815×720,25%](https://github.com/sahanmar/Peony/blob/supporting_images/images/architecture_images/models_1.png)

PeonyBox is a name for a machine learning component of the project. It takes an input from a database, processes it, and then saves an output to the database. In PeonyBox it will be possible train, test and save the models for further usage.

## Visualization

![vizualization|512x512,40%](https://github.com/sahanmar/Peony/blob/supporting_images/images/architecture_images/visualization.png)

Visualization is third part component of this project that takes the data from the database (which were previously uploaded there from PeonyBox) and gives statistics and curves that help to measure quality of models.
