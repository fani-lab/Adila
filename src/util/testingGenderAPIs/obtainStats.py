import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json

barWidth = 0.25

# This script allows you to show the graphs and display the output

'''
ACCURACIES
'''

with open("RetrievedData/genderize_acc.json", 'r') as file_object:  #open the file in write mode
    dataIZE = json.load(file_object) 
    with open("RetrievedData/genderAPI_acc.json", 'r') as file_object:  #open the file in write mode
        dataAPI = json.load(file_object) 

        for i in range(0, 100, 10):
            fig, ax = plt.subplots(figsize =(10, 10))
            dataIZE_temp = dataIZE[i:(i+10)]
            dataAPI_temp = dataAPI[i:(i+10)]

            br1 = np.arange(len(dataIZE_temp))
            br2 = [x + barWidth for x in br1]

            rects1 = ax.bar(br1, dataIZE_temp, color ='r', width = barWidth,
            edgecolor ='grey', label ='Genderize')
            rects2 = ax.bar(br2, dataAPI_temp, color ='g', width = barWidth,
            edgecolor ='grey', label ='GenderAPI')

            ax.set_xlabel("Names")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([50, 110])
            ax.set_xlim([-1, 10])

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.legend()

            plt.savefig(f'Results/accuracies_{i}-{i+10}.png')


        '''
        RESULTS: saved in apiCompared.txt
        '''

        data = pd.read_csv("RetrievedData/output.csv")

        dataIZE_temp = data['genderize'].array
        dataAPI_temp = data['gender-api'].array

        namesF = data['firstName'].array
        namesL = data['lastName'].array


        invalidCount_IZE = 0
        invalidCount_API = 0
        sameCount = 0
        femaleCount = 0

        diff = []

        # Run through all the names and check the following..
        for i in range(0, len(dataAPI_temp)):
            # If Gender-API is NULL
            if(dataAPI_temp[i] == -1):
                invalidCount_API += 1
            # If Genderize is NULL
            if(dataIZE_temp[i] == -1):
                invalidCount_IZE += 1
            # If Both APIs have the same result
            if(dataAPI_temp[i] == dataIZE_temp[i]):
                sameCount += 1
            else:
                # Record name that gave a different result
                diff.append(namesF[i] + " " + namesL[i] + " -> where genderize=" + str(dataIZE_temp[i]) + " and gender-api=" + str(dataAPI_temp[i]))

            if(dataAPI_temp[i] == 1):
                femaleCount += 1


        # Dump Output:
        results = open("Results/apiCompared.txt", "w")

        results.write("RESULTS:\n")
        results.write(f"Same Results for {sameCount} entries\n")

        for x in diff:
            results.write(f"\t{x}\n")

        results.write(f"Invalid Count in Genderize for {invalidCount_IZE} entries\n")
        results.write(f"Invalid Count in GenderAPI for {invalidCount_API} entries\n")


        results.write(f"Female vs Males Names: {femaleCount} vs {100-femaleCount}")
