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
            


    def exportResults_toPickle(self, directory):
        self.df.to_pickle(path=directory)
    
    def exportResults_toCSV(self, directory):
        self.df.to_csv(directory)
    
    def importResults(self, directory):
        self.df = pd.read_pickle(directory)

    def printResults(self, head=None):
        if(head):
            print(self.df.head(head))
        else:
            print(self.df)
    
    def confirmSortedAndUnique(self):
        print(f"UNIQUE: {self.df.index.is_unique}")
        print(f"SORTED: {self.df.index.is_monotonic_increasing}")

    def getCount(self):
        print(f"Number of unique names: {self.df.shape[0]}")


if __name__ == "__main__":
    uniqueNames = UniqueName(attribute='Gender')

    # uniqueNames.DBLP_filterNames(jsonFile='../dblp.v12.json', 
    #                              outputFile='src/util/UniqueNames/dblp_correctNames.json',
    #                              errorFile='src/util/UniqueNames/dblp_failed_to_parse.json')    


    # uniqueNames.searchDBLP(jsonFile='src/util/UniqueNames/dblp_correctNames.json')


    # uniqueNames.exportResults_toPickle(directory='src/util/UniqueNames/uniqueNames_filtered.pkl')
    # uniqueNames.exportResults_toCSV(directory='src/util/UniqueNames/uniqueNames_filtered.csv')

    uniqueNames.importResults(directory='src/util/UniqueNames/uniqueNames_filtered.pkl')
    uniqueNames.confirmSortedAndUnique()
    uniqueNames.printResults(20)
    uniqueNames.getCount()


'''
OTHER STUFF:

# # print(self.df.head())

# # print(self.df.loc['Bengio'])

# # self.df.at['Benigo', ['Gender', 'Probability']] = (1, 3)

# # print(self.df.index.is_monotonic)

# curr = self.df.loc['Benigo']
'''