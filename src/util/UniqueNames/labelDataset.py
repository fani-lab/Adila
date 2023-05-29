'''
OVERVIEW:

1. Find unique names in the dblp/imdb
2. Record Unique Names in pandas dataframe (stored in .pkl files) -> import/export functions provided
3. Map unique name to gender 
4. Label DBLP/IMDB Dataset with gender and discard entries that could not find gender 

GENDER VALUES:
true: male
false: female
null: can't deciced (prob: 0.0)
'''

import json
import pandas as pd
import numpy as np
import pickle as pkl

import grequests

class LabelDataset:
    def __init__(self, attribute) -> None:
        self.df = None
        self.attribute = attribute
        self.filterSet = set() # Used when a the entries filtered need to be specified (e.g. used in IMDB for other files)
    
    def extractFirstName_DBLP(self,name):
        # Remove " thing
        name = name.replace('\"', '')

        # If there is a bracket e.g. textA (textB) -> assume textB is the first name
        is_bracket = name.find('(')
        if(is_bracket >= 0): name = name[is_bracket+1:]

        # Look for a space -> if there is no space -> return entire string
        ind = name.find(' ')
        if(ind >= 0): name = name[0:ind]

        is_bracket = name.find(')')

        if(is_bracket >= 0): return name[:is_bracket]

        return name
    

    def extractFirstName_IMDB(self,name):
        # Filter Characters:
        name = name.replace('\"', '')
        name = name.replace('\'', '')
        
        # Ignore names with these special characters
        if('$' in name or '#' in name or '!' in name): return None

        
        ind = name.find(' ')
        if(ind >= 0): 
            # If the first string is initial -> e.g. A.
            if(name[ind-1] == '.' or ind == 1):
                ind2 = name[ind:].find(' ')
                if(ind2 >= 0): return name[ind+1:ind2]
                return None

            # No Initials:
            return name[0:ind]
        
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
                    names.append(self.extractFirstName_DBLP(author['name'])) # Get First Name
            except Exception as e:
                # First Name cannot be obtained...
                errorLines.append(line)
                print(e)
            print(f"Line {i} processed")
            i += 1
            # Every 100k names, remove duplicate entries
            if(len(names) > 100000):
                namesNP = np.hstack((namesNP, np.array(names))) # Combines names list with unique names list
                namesNP = np.unique(ar=namesNP) # Find/sort unique names -> result is sorted
                names = []
        
        namesNP = np.hstack((namesNP, np.array(names))) # Combines names list with unique names list
        namesNP = np.unique(ar=namesNP) # Find/sort unique names -> result is sorted
        self.df = pd.DataFrame(None, index=namesNP, columns=[self.attribute, 'Probability'])

        # Checks if any lines were skipped
        with open('src/util/UniqueNames/error.txt', 'w') as f:
            for error in errorLines:
                f.write(error + '\n')


    # HEADER:
    # nconst	primaryName	gender genderProbability birthYear	deathYear	primaryProfession	knownForTitles
    
    def searchIMDB(self, tsvFile):
        header = None
        names = []
        namesNP = np.array([])
        i = 0 

        outputFile = open('src/util/UniqueNames/imdb_correctNames.tsv', 'w')

        errorFile = open('src/util/UniqueNames/imdb_errorFile.tsv', 'w')

        for line in open(tsvFile, 'r'):
            if(not header): 
                header = line[:-1].split('\t')
                outputFile.write(f"{header[0]}\t{header[1]}\tgender\tgenderProbability\t{header[2]}\t{header[3]}\t{header[4]}\t{header[5]}\n")
                errorFile.write(f"{header[0]}\t{header[1]}\tgender\tgenderProbability\t{header[2]}\t{header[3]}\t{header[4]}\t{header[5]}\n")
                continue
            
            i += 1
            print(f"Processing #{i}")

            line = line[:-1].split('\t')
            gender = None
            prob = None

            roles = line[4].split(',')
            if(roles[0] != ''):
                # If actor/actress is specified -> assume True/False
                if('actor' in roles):
                    gender = True 
                    prob = 1.0
                elif('actress' in roles):
                    gender = False
                    prob = 1.0
            
            if(not prob):
                # Extract First Name:
                name = self.extractFirstName_IMDB(line[1])
                # Name could not obtained:
                if(name == None or len(name) == 1):
                    errorFile.write(f"{line[0]}\t{line[1]}\t{gender}\t{prob}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\n")
                    continue

                names.append(name)
            
            outputFile.write(f"{line[0]}\t{line[1]}\t{gender}\t{prob}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\n")

            if(len(names) > 100000):
                namesNP = np.hstack((namesNP, np.array(names))) # Combines names list with unique names list
                namesNP = np.unique(ar=namesNP) # Find/sort unique names -> result is sorted
                names = []
        
        
        namesNP = np.hstack((namesNP, np.array(names)))
        namesNP = np.unique(ar=namesNP)
        self.df = pd.DataFrame(None, index=namesNP, columns=[self.attribute, 'Probability'])


    # Example: "CestaAmedeo"
    # For DBLP Only
    def checkNamesTogether(self, test_str):
        res = []
        for i in range(0,len(test_str)-1):
            if test_str[i].islower() and test_str[i+1].isupper():
                return i
        
        return -1

    # Swaps Order based on index (used to deal with cases like "CestaAmedeo")
    # DBLP Only
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
    
            

    # Scans through the result file and updates the dataframe with the appropriate gender for each unique name
    def addGenderResultsFromFile(self, folderOfResults, numOfEntries, inc=1000, start=0):
        # inc: the range for each file (either 500 or 1000 usually)
        # start: where to start loop (usually 0)
        # numOfEntries: end of loop
        
        for i in range(start, numOfEntries-inc, inc):   
            for line in open(f"{folderOfResults}/apiOutput_{i}_to_{(i+inc)}.txt", 'r'):
                results = json.loads(line)
                # Checks if gender was found
                if(results['gender']):
                    # Adds either True/False based on result in dataframe
                    self.df.loc[results['name'], 'Gender'] = results['gender'] == 'male'
                    # Adds Probability value:
                    self.df.loc[results['name'], 'Probability'] = results['probability']
                print(f"{i} to {i+inc} has been searched")
        
        



    # Private method used in makeParallelAPIReqs
    def exception_handler(request, exception):
        print("Request failed")
        

    # Using the grequests library, each name from the dataframe will make a request for the gender information
    # NOTE: Please specify a range to make the API requests
    # For my machine, I had no problems using a range of 1000 for every call
    # The range is [a,b) -> inclusive a, exclusive b

    def makeParallelAPIReqs(self, apiKeyDirectory, a, b):
        key = ""
        rawOutput = open(f'src/util/UniqueNames/IMDBResults/ApiResults/apiOutput_{a}_to_{b}.txt', 'w')
        resultCodes = open(f'src/util/UniqueNames/IMDBResults/ApiResults/resultCodes_{a}_to_{b}.txt', 'w')

        with open(apiKeyDirectory, 'r') as f:
            key = f.readline()

        urls = []
        # creates url string for all the names
        for index in self.df.index[a:b]:
            if(len(index) > 1): 
                urls.append(f"https://api.genderize.io?name={index}&apikey={key}")

        print(f"{len(urls)} URLs Made")
        reqs = (grequests.get(u) for u in urls) # creates url objects for each url
        print("Request Objects made")

        # Makes the requests and writes the results to the file
        for result in grequests.map(reqs, exception_handler=self.exception_handler):
            resultCodes.write(f"{result.status_code} -> {result.url}\n")
            print(f"{result.status_code} -> {result.url}")
            rawOutput.write(f"{result.text}\n")



    # Export to Pickle
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
    

    # Labels IMDB name.basics.tsv file
    def labelIMDB_gender(self, input_tsvFile, output_tsvFile, error_tsvFile):
        header = None
        i = 0 

        outputFile = open(output_tsvFile, 'w')

        errorFile = open(error_tsvFile, 'w')

        # read each line in the names.basics.tsv file
        for line in open(input_tsvFile, 'r'):
            if(not header): 
                header = True
                outputFile.write(line)
                continue
            
            lineArray = line[:-1].split('\t')

            if(lineArray[2] != "None"):
                outputFile.write(line)
                i += 1
            else:
                # Get the first name of the entry
                name = self.extractFirstName_DBLP(lineArray[1])
                if(name.lower() in self.df.index): # lower in case the lower case of the name exists
                    name = name.lower()
                    
                info = self.getDataFromName(name)
                if(info == None): # NULL gender -> add to filter set and print to error file
                    errorFile.write(f"{lineArray}\n")
                    self.filterSet.add(lineArray[0]) # Filterset will be used to remove entries from the other files
                else:
                    i += 1
                    # Prints out successful entries with their gender displayed 
                    outputFile.write(f"{lineArray[0]}\t{lineArray[1]}\t{info[0]}\t{info[1]}\t{lineArray[2]}\t{lineArray[3]}\t{lineArray[4]}\t{lineArray[5]}\n")
            
            print(f"{i} successful experts...")


    # Removes entries for invalid members from the title.principals.tsv and title.basics.tsv files
    def removeTitlesWithNULLGender(self, dir_titleBasics, dir_titlePrincipals, newDir_titleBasics, newDir_titlePrincipals):
        # The non labelled files
        titlePrincipals = open(dir_titlePrincipals, "r") 
        titleBasics = open(dir_titleBasics, "r")

        # The filtered versions will go here
        outTitlePrincipals = open(newDir_titlePrincipals, "w")
        outTitleBasics = open(newDir_titleBasics, "w")

        # Get header and first line of titlebasics
        line_titleBasics = titleBasics.readline()
        outTitleBasics.write(line_titleBasics)
        line_titleBasics = titleBasics.readline()

        # Get header and first line
        line_titlePrincipals = titlePrincipals.readline()
        outTitlePrincipals.write(line_titlePrincipals)
        line_titlePrincipals = titlePrincipals.readline()

        # Loop through all the titles in title.basics.tsv
        while(line_titleBasics != ""):
            removeEntry = False
            entry_titleBasics = line_titleBasics
            entry_titlePrincipals = []

            lineArray_titleBasics = line_titleBasics[:-1].split('\t')
            lineArray_titlePrincipals = line_titlePrincipals[:-1].split('\t')

            print(f"Title: {lineArray_titleBasics[0]}")

            # Checks if the title has no members insides
            if(lineArray_titleBasics[0] != lineArray_titlePrincipals[0]):
                if(int(lineArray_titleBasics[0][2:]) > int(lineArray_titlePrincipals[0][2:])):
                    line_titlePrincipals = titlePrincipals.readline()
                else:
                    line_titleBasics = titleBasics.readline()
                continue
            else:    
                # Loop through all the members of a given title 
                while(lineArray_titleBasics[0] == lineArray_titlePrincipals[0]):
                    entry_titlePrincipals.append(line_titlePrincipals)
                    # Don't include title if the member's gender cannot be found:
                    if(lineArray_titlePrincipals[2] in self.filterSet): 
                        removeEntry = True

                    line_titlePrincipals = titlePrincipals.readline()
                    if(line_titlePrincipals == ""): break
                    lineArray_titlePrincipals = line_titlePrincipals[:-1].split('\t')

            if(not removeEntry):
                # Include Title if the gender is included for all members
                outTitlePrincipals.write("".join(entry_titlePrincipals))
                outTitleBasics.write(entry_titleBasics)
            
            line_titleBasics = titleBasics.readline()

        titlePrincipals.close()
        titleBasics.close()
        outTitleBasics.close()
        outTitlePrincipals.close()




    # Labels the dataframe 
    def labelDataset_gender(self, datasetDirectory, newDatasetDirectory):
        newDataset = open(newDatasetDirectory, 'w')

        firstOutput = True
        
        checked = 0
        success = 0

        for line in open(datasetDirectory, 'r'):

            # A flag value that goes to TRUE if one of the author's gender could not be found
            failed = False
            if line[0] == '[': 
                newDataset.write('[' + '\n')
                continue
            if line[0] == ']': 
                newDataset.write('[' + '\n')
                continue
            
            if(line[0] == ','):
                line = line[1:]
            
            row = json.loads(line)
            for i in range(0, len(row['authors'])):
                name = self.extractFirstName_DBLP(row['authors'][i]['name'])
                if(name.lower() in self.df.index): 
                    name = name.lower()
                    
                info = self.getDataFromName(name)
                if(info == None):
                    failed = True
                else:
                    row['authors'][i]['gender'] = {'value': info[0], 'probability': info[1]} 
                

            if(not failed): 
                if(not firstOutput): newDataset.write(',')
                firstOutput = False 
                newDataset.write(str(json.dumps(row)) + '\n')
                success += 1

            checked += 1
            print(f"---\nSuccess: {success}\nChecked: {checked}")
        newDataset.close()




    # Return a tuple: (gender, probability)
    # returns None if gender could not be found
    def getDataFromName(self, name):
        try:
            if(self.df.loc[name,self.attribute] == True):
                return ('M', self.df.loc[name, "Probability"])
            elif(self.df.loc[name,self.attribute] == False):
                return ('F', self.df.loc[name, "Probability"])


            return None
        except KeyError:
            return None
    
    def confirmSortedAndUnique(self):
        print(f"UNIQUE: {self.df.index.is_unique}")
        print(f"SORTED: {self.df.index.is_monotonic_increasing}")

    def getCount(self):
        print(f"Number of unique names: {self.df.shape[0]}")



def runIMDB(labelIMDB: LabelDataset):
    labelIMDB.searchIMDB('../name.basics.tsv')
    labelIMDB.exportResults_toPickle(directory='src/util/UniqueNames/IMDBResults/uniqueNames.pkl')
    labelIMDB.exportResults_toCSV(directory='src/util/UniqueNames/IMDBResults/uniqueNames.csv')

    # Part 2: Make API Requests:


    for i in range(0, 447000, 500):
        print(f"Working on {i} to {i+500}")
        labelIMDB.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=i, b=(i+500))

    # # Add Remainings
    labelIMDB.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=447000, b=447082)

    # Part 3: Fill in Dataframe with Gender Results:

    labelIMDB.addGenderResultsFromFile(folderOfResults="src/util/UniqueNames/IMDBResults/ApiResults", numOfEntries=5000, start=0, inc=1000)
    labelIMDB.addGenderResultsFromFile(folderOfResults="src/util/UniqueNames/IMDBResults/ApiResults", numOfEntries=447000, start=5000, inc=500)
    labelIMDB.addGenderResultsFromFile(folderOfResults="src/util/UniqueNames/IMDBResults/ApiResults", numOfEntries=447082, start=447000, inc=82)

    labelIMDB.exportResults_toPickle(directory='src/util/UniqueNames/IMDBResults/uniqueNames.pkl')
    labelIMDB.exportResults_toCSV(directory='src/util/UniqueNames/IMDBResults/uniqueNames.csv')

    labelIMDB.exportResults_toPickle(directory='src/util/UniqueNames/IMDBResults/uniqueNames_labelled.pkl')
    labelIMDB.exportResults_toCSV(directory='src/util/UniqueNames/IMDBResults/uniqueNames_labelled.csv')

    # Part 4: Test Dataframe:

    print(labelIMDB.getDataFromName('Ülle')) # Female 
    print(labelIMDB.getDataFromName('Émélie')) # Female
    print(labelIMDB.getDataFromName('Yvetot')) # Male
    print(labelIMDB.getDataFromName('Gabriel')) # Male


    # Part 5: Label Dataset:
    labelIMDB.labelIMDB_gender(input_tsvFile='src/util/UniqueNames/imdb_correctNames.tsv', output_tsvFile='../name.basics_labelled.tsv', error_tsvFile='../name.basics_failedToLabel.txt')
    labelIMDB.removeTitlesWithNULLGender(dir_titleBasics="../title.basics.tsv",
                                           dir_titlePrincipals="../title.principals.tsv",
                                           newDir_titleBasics="../title.basics_labelled.tsv",
                                           newDir_titlePrincipals="../title.principals_labelled.tsv")


def runDBLP(labelDBLP: LabelDataset):
    # STEP 1: Filter Entries and Search for Names in DBLP:
    labelDBLP.DBLP_filterNames(jsonFile='../dblp.v12.json', 
                                outputFile='src/util/UniqueNames/dblp_correctNames.json',
                                errorFile='src/util/UniqueNames/dblp_failed_to_parse.json')    

    labelDBLP.searchDBLP(jsonFile='src/util/UniqueNames/dblp_correctNames.json')

    labelDBLP.exportResults_toPickle(directory='src/util/UniqueNames/uniqueNames_filtered.pkl')
    labelDBLP.exportResults_toCSV(directory='src/util/UniqueNames/uniqueNames_filtered.csv')


    # STEP 2: Complete API Requests

    # API Requests for First Part: 
    for i in range(0, 273000, 1000):
        print(f"Working on {i} to {i+1000}")
        labelDBLP.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=i, b=(i+1000))

    # Add Remainings
    labelDBLP.makeParallelAPIReqs(apiKeyDirectory="src/util/UniqueNames/secretKey.txt", a=274000, b=275860)

    # STEP 3: Extract API Results Raw Output to Dataframe
    labelDBLP.addGenderResultsFromFile(folderOfResults="src/util/UniqueNames/Results", numOfEntries=275000)
    
    labelDBLP.exportResults_toCSV(directory='src/util/UniqueNames/uniqueNames_populated.csv')
    labelDBLP.exportResults_toPickle(directory='src/util/UniqueNames/uniqueNames_populated.pkl')

    # Test dataframe with some names
    print(labelDBLP.getDataFromName('Pranava'))
    print(labelDBLP.getDataFromName('Suibo'))
    print(labelDBLP.getDataFromName('Hamed'))
    print(labelDBLP.getDataFromName('Julia'))

    # STEP 4: Label Dataset
    labelDBLP.labelDataset_gender(datasetDirectory='src/util/UniqueNames/dblp_correctNames.json', newDatasetDirectory='src/util/UniqueNames/dblp_labelledGender.json')



if __name__ == "__main__":
    # runDBLP(LabelDataset(attribute='Gender'))
    # runIMDB(LabelDataset(attribute='Gender'))
    
    ld_dblp = LabelDataset(attribute='Gender')
    ld_imdb = LabelDataset(attribute='Gender')

    ld_dblp.importResults("src/util/UniqueNames/uniqueNames_filtered.pkl")
    
    print(ld_dblp.getDataFromName('Luis'))


