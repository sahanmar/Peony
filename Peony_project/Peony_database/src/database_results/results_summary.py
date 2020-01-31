from PeonyPackage.PeonyDb import MongoDb
from typing import Dict


class PeonyDbResults:
    def __init__(self):
        self.api = MongoDb()
        self.data = self.api.get_model_results(filter_dict={})

    def structurize_data(self) -> Dict[str, Dict["str", dict]]:
        structurized_data: Dict[str, Dict["str", dict]] = {}
        for record in self.data:

            model = record["model"]
            category_1 = record["category_1"]
            category_2 = record["category_2"]
            dataset = f"{category_1} / {category_2}"
            acquisition_function = record["acquisition_function"]

            if model not in structurized_data:
                structurized_data[model] = {dataset: {acquisition_function: record}}
            else:
                if dataset not in structurized_data[model]:
                    structurized_data[model][dataset] = {acquisition_function: record}
                else:
                    if acquisition_function not in structurized_data[model][dataset]:
                        structurized_data[model][dataset][acquisition_function] = record
                    else:
                        structurized_data[model][dataset][acquisition_function][
                            "results"
                        ] = (
                            structurized_data[model][dataset][acquisition_function][
                                "results"
                            ]
                            + record["results"]
                        )
        return structurized_data


if __name__ == "__main__":
    results = PeonyDbResults()
    structurized_results = results.structurize_data()
    print("finished")
