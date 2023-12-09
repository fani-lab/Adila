from __future__ import annotations
import pandas as pd
import os

# /output/dblp/bnn/{long name}/rerank/gender/dp/{file}
# f0.test.pred.det_cons.100.ndkl.faireval.csv

def obtainResults(algorithm: str, folds: int, folder_dir: str) -> None:
    """
    Obtains results from a specific folder which has the following in the path: dataset, baseline, metric, and eo/dp
    Will calculate average among folds and then output to csv file in the same directory as {algorithm}.100.faireval.results.csv and {algorithm}.100.utileval.results.csv
    Args:
        algorithm: name of algorithm
        folds: number of folds in the baseline (3 for random, 5 for bnn and bnn_emb)
        folder_dir: folder the specfic set of results
    Returns:
        None
    """
    outputs_fairness = {
               "Color Blind (before) NDKL" : [],  # Fairness Metric
               "NDKL After": [],  # Fairness Metric
               "Skew before(p/np)": [], # Fairness Metric
               "Skew after(p/np)": [],  # Fairness Metric
    }

    outputs_utility = {           
               "Color Blind MAP10": [],  # Utility Metric
               "MAP10 After": [], # Utility Metric
               "Color Blind NDCG10": [], # Utility Metric
               "NDCG10 After": []  # Utility Metric
    }

    errors = []

    # Obtain NDKL Results:
    result = [0,0]

    for i in range(0,folds):
        try:
            filename = os.path.join(folder_dir, f"f{i}.test.pred.{algorithm}.100.ndkl.faireval.csv")
            df = pd.read_csv(filename)
            result[0] += df.iat[0,1]
            result[1] += df.iat[0,2]
        except:
            errors.append(f"Could not find nkdl file for {algorithm} -> f{i}\n")


    outputs_fairness["Color Blind (before) NDKL"].append(result[0] / folds)
    outputs_fairness["NDKL After"].append(result[1] / folds)

    # Obtain Skew Results:
    result = [0,0,0,0]

    for i in range(0,folds):
        try:
            filename = os.path.join(folder_dir, f"f{i}.test.pred.{algorithm}.100.skew.faireval.csv")
            df = pd.read_csv(filename)
            result[0] += df.iat[0,1]
            result[1] += df.iat[0,2]
            result[2] += df.iat[1,1]
            result[3] += df.iat[1,2]

        except:
            errors.append(f"Could not find skew file for {algorithm} -> f{i}\n")

    
    outputs_fairness["Skew before(p/np)"].append(f"p: {result[0] / folds} np: {result[2] / folds}")
    outputs_fairness["Skew after(p/np)"].append(f"p: {result[1] / folds} np: {result[3] / folds}")


    # Obtain MAP10 and NDCG Results:
    result = [0,0,0,0]

    for i in range(0,folds):
        try:
            filename = os.path.join(folder_dir, f"f{i}.test.pred.{algorithm}.100.utileval.csv")
            df = pd.read_csv(filename)
            row_ndcg = (df[df['metric'] == 'ndcg_cut_10'].index)[0]
            row_map = (df[df['metric'] == 'map_cut_10'].index)[0]
            result[0] += df.iat[row_ndcg, 1]
            result[1] += df.iat[row_ndcg, 2]
            result[2] += df.iat[row_map, 1]
            result[3] += df.iat[row_map, 2]
        except:
            errors.append(f"Could not find MAP10 & NDCG results for {algorithm} -> f{i}\n")


    outputs_utility["Color Blind MAP10"].append(result[2] / folds)
    outputs_utility["MAP10 After"].append(result[3] / folds)

    outputs_utility["Color Blind NDCG10"].append(result[0] / folds)
    outputs_utility["NDCG10 After"].append(result[1] / folds)


    if(errors): raise Exception("".join(errors))

    df_fairness = pd.DataFrame(data=outputs_fairness)
    df_fairness.to_csv(folder_dir + f"{algorithm}.100.faireval.results.csv")

    df_utility = pd.DataFrame(data=outputs_utility)
    df_utility.to_csv(folder_dir + f"{algorithm}.100.utileval.results.csv")


def obtain_results_all_subfolders() -> None:
    for dataset in ["dblp", "imdb"]:
        for baseline in ["bnn", "bnn_emb", "random"]:
            for metric in ["gender", "popularity"]:
                folder = f"output/Adila0.2.0/{dataset}/{baseline}"
                if(not os.path.isdir(folder)):
                    print(f"Could not find {folder}")
                    continue
                for big_name in os.listdir(folder): # big_name is the very long name in the folder path
                    if(os.path.isdir(folder + f"/{big_name}")):
                        for y in ["dp", "eo"]:
                            folder2 = folder + f"/{big_name}/rerank/{metric}/{y}/"
                            algorithms = ["det_greedy", "det_relaxed", "det_cons", "fa-ir.1", "fa-ir.5", "fa-ir.10"]
                            if(os.path.isdir(folder2)):
                                for algorithm in algorithms:
                                    try:
                                        if(baseline == "random"):
                                            obtainResults(algorithm=algorithm, folds=3, folder_dir=folder2)
                                        else:
                                            obtainResults(algorithm=algorithm, folds=5, folder_dir=folder2)
                                    except Exception as e:
                                        print("\tErrors with " + folder2 + ", " + algorithm)
                                        print(e)
                            else:
                                print(f"Could not find {folder2}")


if __name__ == "__main__":
    obtain_results_all_subfolders() 

   


# OLD VERSION: 

# from __future__ import annotations
# import pandas as pd
# import os

# def obtainResults(baselines: dict[str, int], algorithms: list[str], folderDir: str, outputDir: str):
#     """
#     Function to generate a csv file with results from a specfic set of outputs
#     Expected Files (For 1 algorithm):
#         i: number of folds in the baseline

#         folderDir
#         ├── baselineName 
#         │   ├── f{i}.test.pred.{algorithmName}.100.ndkl.faireval.csv
#         │   ├── f{i}.test.pred.{algorithmName}.100.skew.faireval.csv
#         │   ├── f{i}.test.pred.{algorithmName}.100.utileval.csv

#     Expected File Structure (For 2+ Alg):
#         i: number of folds in the baseline

#         folderDir
#         ├── algorithmName
#         │    ├── baselineName 
#         │    │   ├── f{i}.test.pred.{algorithmName}.100.ndkl.faireval.csv
#         │    │   ├── f{i}.test.pred.{algorithmName}.100.skew.faireval.csv
#         │    │   ├── f{i}.test.pred.{algorithmName}.100.utileval.csv

#     Args:
#         baselines: dictionary which includes all baselines in format: {baselineName: numOfFolds}
#         algorithm: name of algorithm
#         folderDir: folder the specfic set of results
#         outputDir: location to place output csv file
#     """

#     outputs = {"Baseline": [],
#                "Algorithm": [],
#                "Color Blind (before) NDKL" : [], 
#                "NDKL After": [], 
#                "Skew before(p/np)": [],
#                "Skew after(p/np)": [], 
#                "Color Blind MAP10": [], 
#                "MAP10 After": [], 
#                "Color Blind NDCG10": [], 
#                "NDCG10 After": [] 
#                }

#     errors = []

#     for baselineName in baselines:
#         folds = baselines[baselineName]
#         for alg in algorithms:
#             outputs["Algorithm"].append(alg)
#             outputs["Baseline"].append(baselineName)

#             # Go to directory for specific baseline and algorithm
#             fileBase = folderDir
#             if(len(algorithms) > 1): fileBase = os.path.join(fileBase, f"{alg}")
#             fileBase = os.path.join(fileBase, f"{baselineName}")

            
#             # Obtain NDKL Results:
#             result = [0,0]

#             for i in range(0,folds):
#                 try:
#                     filename = os.path.join(fileBase, f"f{i}.test.pred.{alg}.100.ndkl.faireval.csv")
#                     df = pd.read_csv(filename)
#                     result[0] += df.iat[0,1]
#                     result[1] += df.iat[0,2]
#                 except:
#                     errors.append(f"Could not find nkdl file for {alg}, {baselineName} -> f{i}\n")


            
#             outputs["Color Blind (before) NDKL"].append(result[0] / folds)
#             outputs["NDKL After"].append(result[1] / folds)


#             # Obtain Skew Results:
#             result = [0,0,0,0]

#             for i in range(0,folds):
#                 try:
#                     filename = os.path.join(fileBase, f"f{i}.test.pred.{alg}.100.skew.faireval.csv")
#                     df = pd.read_csv(filename)
#                     result[0] += df.iat[0,1]
#                     result[1] += df.iat[0,2]
#                     result[2] += df.iat[1,1]
#                     result[3] += df.iat[1,2]

#                 except:
#                     errors.append(f"Could not find skew file for {alg}, {baselineName} -> f{i}\n")

            
#             outputs["Skew before(p/np)"].append(f"p: {result[0] / folds} np: {result[2] / folds}")
#             outputs["Skew after(p/np)"].append(f"p: {result[1] / folds} np: {result[3] / folds}")


#             # Obtain MAP10 and NDCG Results:
#             result = [0,0,0,0]

#             for i in range(0,folds):
#                 try:
#                     filename = os.path.join(fileBase, f"f{i}.test.pred.{alg}.100.utileval.csv")
#                     df = pd.read_csv(filename)
#                     row_ndcg = (df[df['metric'] == 'ndcg_cut_10'].index)[0]
#                     row_map = (df[df['metric'] == 'map_cut_10'].index)[0]
#                     result[0] += df.iat[row_ndcg, 1]
#                     result[1] += df.iat[row_ndcg, 2]
#                     result[2] += df.iat[row_map, 1]
#                     result[3] += df.iat[row_map, 2]
#                 except:
#                     errors.append(f"Could not find MAP10 & NDCG results for {alg}, {baselineName} -> f{i}\n")


#             outputs["Color Blind MAP10"].append(result[2] / folds)
#             outputs["MAP10 After"].append(result[3] / folds)

#             outputs["Color Blind NDCG10"].append(result[0] / folds)
#             outputs["NDCG10 After"].append(result[1] / folds)


#     if(errors): raise Exception("".join(errors))
    
#     df = pd.DataFrame(data=outputs)

#     df.to_csv(outputDir + f"{folderDir[folderDir.rindex('/'):]}.csv")



# def obtainResults_fa_ir(folderDir: str, outputDir: str):
#     """
#     Function to generate a csv file with results from folder of results
#     Each folder has it's own dataset, fairness notion, k, and significance level
   
#     Args:
#         folderDir: folder the specfic set of results
#         outputDir: location to place output csv file
#     """
#     obtainResults({"random": 3, "bnn": 5, "bnn_emb": 5}, ["fa-ir"], folderDir, outputDir)

# def obtainResults_det(folderDir: str, outputDir: str):
#     """
#     Function to generate a file with use
#     Each folder has it's own dataset, fairness notion, and k

#     Args:
#         folderDir: folder the specfic set of results
#         outputDir: location to place output csv file
#     """
#     obtainResults({"random": 3, "bnn": 5, "bnn_emb": 5}, ["det_cons", "det_greedy", "det_relaxed"], folderDir, outputDir)



# if __name__ == "__main__":
    # obtainResults_fa_ir("output/dblp.v12.json/DemographicParity.gender.0.1.100.fa-ir.imdb", "output/dblp.v12.json")
    # obtainResults_det("output/dblp.v12.json/DemographicParity.popularity.100.det.dblp", "output/dblp.v12.json")
