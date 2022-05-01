import numpy as np

from PeonyPackage.PeonyDb import MongoDb
from tabulate import tabulate

def main():
    api = MongoDb()

    results = list(api.database["models_results"].find()) 

    tabulate_list = []
    header = ["mongo_id", "run_name"] + list(range(1,11))

    batch_size = 0

    for result in results:
        
        if batch_size == 0:
            batch_size = int(result["learning_step"])
        if batch_size != int(result["learning_step"]):
            tabulate_list.append(["" for i in range(11)])
            batch_size = int(result["learning_step"])

        name = f"{result['model']}_{result['acquisition_function']}_{result['learning_step']}_{result['active_learning_iterations']}_{result['category_1']}_{result['category_2']}"
        mean = np.mean(result["results"],axis = 0)
        std = np.std(result["results"],axis = 0)

        res_id = result["_id"]

        results_list = [res_id, name]
        i_sampler = len(mean)//10

        for i, (m, s) in enumerate(zip(mean, std)):
            if (i+1) % i_sampler == 0:
                results_list.append(f"{round(m[0],3)}±{round(s[0],3)}")

        tabulate_list.append(results_list)

    print(tabulate(tabulate_list, headers=header, tablefmt='orgtbl'))

if __name__=="__main__":
    main()
