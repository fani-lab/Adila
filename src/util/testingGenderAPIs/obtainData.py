import pandas as pd

import requests, json

# This script extracts the data from the two APIs

data = pd.read_csv("RetrievedData/input.csv")

header = "firstName,lastName,gender-api,genderize"

output = {'firstName': [], 'lastName': [], 'gender-api': [], 'genderize': []}

# A list of all the accuracies -> these are used in obtainStats.py to be compared
accuracies_genderize = []
accuracies_genderAPI = []

# The log files contain the raw output from the apis
log_A = open("RetrievedData/genderizeLOG.txt", "w")
log_B = open("RetrievedData/genderAPILOG.txt", "w")


for i in range(0,100):
    # Genderize API
    req = requests.get("https://api.genderize.io?name=" + data['firstName'][i])
    results = json.loads(req.text)
    output['firstName'].append(data['firstName'][i])
    output['lastName'].append(data['lastName'][i])

    log_A.write(req.text + ",\n")

    '''
    0 = Male
    1 = Female
    -1 = NULL
    '''
    if(results['gender'] == "female"): 
        output['genderize'].append(1)
    elif(results['gender'] == "male"):
        output['genderize'].append(0)
    else:
        output['genderize'].append(-1)
    

    accuracies_genderize.append(results['probability']*100)

    # Gender-API
    req = requests.get("https://gender-api.com/get?split=" + data['firstName'][i] + "%20" + data['lastName'][i] + "&key=ddpwJPnQqdP3otFMPz8ZppaA4SBsqRRnVlSK")
    
    log_B.write(req.text + ",\n")

    results_G_API = json.loads(req.text)

    accuracies_genderAPI.append(results_G_API['accuracy'])

    if(results_G_API['gender'] == "female"): 
        output['gender-api'].append(1)
    elif(results_G_API['gender'] == "male"):
        output['gender-api'].append(0)
    else:
        output['gender-api'].append(-1)



# Dump data onto files:
dataFrame = pd.DataFrame(output)

dataFrame.to_csv("RetrievedData/output.csv")

with open("RetrievedData/genderize_acc.json", 'w') as file_object:  #open the file in write mode
    json.dump(accuracies_genderize, file_object)

with open("RetrievedData/genderAPI_acc.json", 'w') as file_object:  #open the file in write mode
    json.dump(accuracies_genderAPI, file_object)


log_A.close()
log_B.close()