'''
OVERVIEW:

1. Find unique names in the dblp
2. Record Unique Names somewhere (csv, pkl, dict, etc)
3. Map unique name to gender 

GENDER VALUES:
true: male
false: female
null: can't deciced (prob: 0.0)
'''

import json
import pandas as pd
import numpy as np
import pickle as pkl

# import requests
import grequests

class UniqueName:
    def __init__(self, attribute) -> None:
        self.df = None
        self.attribute = attribute

    def extractFirstName(self,name):
        # Remove " thing
        name = name.replace('\"', '')

        is_bracket = name.find('(')
        if(is_bracket >= 0): name = name[is_bracket+1:]

        ind = name.find(' ')
        if(ind >= 0): name = name[0:ind]

        is_bracket = name.find(')')

        if(is_bracket >= 0): return name[:is_bracket]

        return name

    def searchDBLP(self, jsonFile) -> None:
        names = []
        i = 0
        errorLines = []
        namesNP = np.array([])

        for line in open(jsonFile, 'r'):
            if(line[0] == ','):
                line = line[1:]
            try:
                row = json.loads(line)
                for author in row['authors']:
                    names.append(self.extractFirstName(author['name']))
            except Exception as e:
                errorLines.append(line)
                print(e)
            print(f"Line {i} processed")
            i += 1
            if(len(names) > 100000):
                namesNP = np.hstack((namesNP, np.array(names)))
                namesNP = np.unique(ar=namesNP)
                names = []
        
        
        namesNP = np.hstack((namesNP, np.array(names)))
        namesNP = np.unique(ar=namesNP)
        self.df = pd.DataFrame(None, index=namesNP, columns=[self.attribute, 'Probability'])

        # Checks if any lines were skipped
        with open('src/util/UniqueNames/error.txt', 'w') as f:
            for error in errorLines:
                f.write(error + '\n')

    # Example: "CestaAmedeo"
    def checkNamesTogether(self, test_str):
        res = []
        for i in range(0,len(test_str)-1):
            if test_str[i].islower() and test_str[i+1].isupper():
                return i
        
        return -1

    def swapNames(self, test_str, switchInd):
        for i in range(switchInd, -1, -1):
            if(test_str[i] == ' '):
                newStr = test_str[switchInd+1:] + ' ' + test_str[i+1: switchInd+1]
                return newStr
        
        newStr = test_str[switchInd+1:] + ' ' + test_str[0: switchInd+1]
        return newStr


    # If we find a name with case 1: 
    def DBLP_filterNames(self, jsonFile, outputFile, errorFile) -> None:
        processedCount = 0
        firstOutput = False
        firstError = False

        outputFile = open(outputFile, 'w')
        outputFile.write('[\n')

        errorFile = open(errorFile, 'w')
        errorFile.write('[\n')

        for line in open(jsonFile, 'r'):
            errorFound = False
            if(line[0] == ','):
                line = line[1:]

            if(line[0] != '[' and line[0] != ']'):
                row = json.loads(line)
                if('authors' not in row): continue
                for i in range(0, len(row['authors'])):
                    tempName = row['authors'][i]['name']
                    x = self.checkNamesTogether(tempName)
                    if(x != -1):
                        row['authors'][i]['name'] = self.swapNames(tempName, x)
                    elif(tempName[1] == ' ' or tempName[1] == '.' or tempName[1] == '-' or (not tempName[0].isalpha())):  
                        if(firstError): 
                            errorFile.write(',')
                        else:
                            firstError = True
                        errorFile.write(str(json.dumps(row)) + '\n')
                        # print(f'Error on name: {tempName}')
                        errorFound = True
                
                if(not errorFound):
                    if(firstOutput): 
                        outputFile.write(',')
                    else:
                        firstOutput = True
                    outputFile.write(str(json.dumps(row)) + '\n')
                print(f'Proccessed {processedCount}')
                processedCount += 1
        

        outputFile.write(']')
        errorFile.write(']')
            

    def addGenderResultsFromFile(self, folderOfResults, numOfEntries):
        x = 0 
        for i in range(0, numOfEntries-1000, 1000):   
            for line in open(f"{folderOfResults}/apiOutput_{i}_to_{(i+1000)}.txt", 'r'):
                results = json.loads(line)
                if(results['gender']):
                    self.df.loc[results['name'], 'Gender'] = results['gender'] == 'male'
                    self.df.loc[results['name'], 'Probability'] = results['probability']
                x += 1
        
        print(f"{x} names have been searched")



    def exception_handler(request, exception):
        print("Request failed")
        
    # [a,b)
    def makeParallelAPIReqs(self, apiKeyDirectory, a, b):
        key = ""
        rawOutput = open(f'src/util/UniqueNames/Results/apiOutput_{a}_to_{b}.txt', 'w')
        resultCodes = open(f'src/util/UniqueNames/Results/resultCodes_{a}_to_{b}.txt', 'w')

        with open(apiKeyDirectory, 'r') as f:
            key = f.readline()

        urls = []
        for index in self.df.index[a:b]:
            if(len(index) > 1): 
                urls.append(f"https://api.genderize.io?name={index}&apikey={key}")

        print(f"{len(urls)} URLs Made")
        reqs = (grequests.get(u) for u in urls)
        print("Request Objects made")

        for result in grequests.map(reqs, exception_handler=self.exception_handler):
            resultCodes.write(f"{result.status_code} -> {result.url}\n")
            print(f"{result.status_code} -> {result.url}")
            rawOutput.write(f"{result.text}\n")

    def exportResults_toPickle(self, directory):
        self.df.to_pickle(path=directory)
    
    def exportResults_toCSV(self, directory):
        self.df.to_csv(directory)
    
    def importResults(self, directory):
        self.df = pd.read_pickle(directory)
    
    def importResults_csv(self, directory):
        self.df = pd.read_csv(directory)
        names = []
        for name in self.df['name']: names.append(name)

        namesNP = np.unique(ar=np.array(names))
        self.df = pd.DataFrame(None, index=namesNP, columns=[self.attribute, 'Probability'])


    def printResults(self, head=None):
        if(head):
            print(self.df.head(head))
        else:
            print(self.df)
    
    def getDataFromName(self, name):
        return (self.df.loc[name, self.attribute], self.df.loc[name, "Probability"])
    
    def confirmSortedAndUnique(self):
        print(f"UNIQUE: {self.df.index.is_unique}")
        print(f"SORTED: {self.df.index.is_monotonic_increasing}")

    def getCount(self):
        print(f"Number of unique names: {self.df.shape[0]}")



def extractData():
    # Obtain Dataframe
    uniqueNames.importResults_csv(directory='src/util/UniqueNames/uniqueNames_filtered.csv')

    # API Requests for First Part: 
    for i in range(0, 273000, 1000):
        print(f"Working on {i} to {i+1000}")
        uniqueNames.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=i, b=(i+1000))


    # Add Remainings
    uniqueNames.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=274000, b=275860)

    # Extract from Files
    uniqueNames.addGenderResultsFromFile(folderOfResults="src/util/UniqueNames/Results", numOfEntries=275000)
    
    # Print
    uniqueNames.exportResults_toCSV(directory='src/util/UniqueNames/uniqueNames_populated.csv')
    uniqueNames.exportResults_toPickle(directory='src/util/UniqueNames/uniqueNames_populated.pkl')

if __name__ == "__main__":
    uniqueNames = UniqueName(attribute='Gender')

    

    # uniqueNames.DBLP_filterNames(jsonFile='../dblp.v12.json', 
    #                              outputFile='src/util/UniqueNames/dblp_correctNames.json',
    #                              errorFile='src/util/UniqueNames/dblp_failed_to_parse.json')    


    # uniqueNames.searchDBLP(jsonFile='src/util/UniqueNames/dblp_correctNames.json')


    # uniqueNames.exportResults_toPickle(directory='src/util/UniqueNames/uniqueNames_filtered.pkl')
    # uniqueNames.exportResults_toCSV(directory='src/util/UniqueNames/uniqueNames_filtered.csv')

    # uniqueNames.importResults(directory='src/util/UniqueNames/uniqueNames_filtered.pkl')

    # extractData()

    '''
    USE THE POPULATED DATAFRAME:
    '''

    uniqueNames.importResults('src/util/UniqueNames/uniqueNames_populated.pkl')

    # Test with some names:
    print(uniqueNames.getDataFromName('justin'))
    print(uniqueNames.getDataFromName('Suibo'))
    print(uniqueNames.getDataFromName('Hamed'))
    print(uniqueNames.getDataFromName('Julia'))


    # uniqueNames.confirmSortedAndUnique()

    # uniqueNames.getCount()


'''
OTHER STUFF:

# # print(self.df.head())

# # print(self.df.loc['Bengio'])

# # self.df.at['Benigo', ['Gender', 'Probability']] = (1, 3)

# # print(self.df.index.is_monotonic)

# curr = self.df.loc['Benigo']

---

FUNCTIONS I USED TO get values (cleaner versions in the class itself for future use)


    def addGenderResults(self, apiKeyDirectory):
        key = ""
        success = 0
        x = 0 
        rawOutput = open('src/util/UniqueNames/apiOutput_169k_to_200k.txt', 'w')
        with open(apiKeyDirectory, 'r') as f:
            key = f.readline()        
        for index in self.df.index:
            if(x < 169000): 
                x += 1
                continue
            if(x >= 200000): 
                x += 1
                continue
            if(len(index) <= 1): continue
            req = requests.get(f"https://api.genderize.io?name={index}&apikey={key}")
            if(not req): break
            results = json.loads(req.text)
            rawOutput.write(req.text + "\n")

            print(f"{x} names have been searched")
            if(results['gender']):
                success += 1
                self.df.loc[index, 'Gender'] = results['gender'] == 'male'
                self.df.loc[index, 'Probability'] = results['probability']
            x += 1
        

        print(f"{success} out of {x} were successful")




    def exception_handler(request, exception):
        print("Request failed")
        
    # [a,b)
    def makeParallelAPIReqs(self, apiKeyDirectory, a, b):
        key = ""
        rawOutput = open(f'src/util/UniqueNames/Results/apiOutput_{a}_to_{b}.txt', 'w')
        resultCodes = open(f'src/util/UniqueNames/Results/resultCodes_{a}_to_{b}.txt', 'w')

        with open(apiKeyDirectory, 'r') as f:
            key = f.readline()

        urls = []
        for index in self.df.index[a:b]:
            if(len(index) > 1): 
                urls.append(f"https://api.genderize.io?name={index}&apikey={key}")

        print(f"{len(urls)} URLs Made")
        reqs = (grequests.get(u) for u in urls)
        print("Request Objects made")

        for result in grequests.map(reqs, exception_handler=self.exception_handler):
            resultCodes.write(f"{result.status_code} -> {result.url}\n")
            print(f"{result.status_code} -> {result.url}")
            rawOutput.write(f"{result.text}\n")

'''