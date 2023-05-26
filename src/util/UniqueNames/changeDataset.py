# Converts the true/false in the dataset to M/F

# M = Male
# F = Female
import json

def convertDBLP(jsonFile : str, outputFile : str):
    processedCount = 0
    firstOutput = False

    outputFile = open(outputFile, 'w')
    outputFile.write('[\n')


    for line in open(jsonFile, 'r'):
        print(processedCount)
        processedCount += 1
        if(line[0] == ','):
            line = line[1:]

        if(line[0] != '[' and line[0] != ']'):
            row = json.loads(line)
            if('authors' not in row): continue
            for i in range(0, len(row['authors'])):
                currValue = row['authors'][i]['gender']['value']
                if(currValue == True):
                    row['authors'][i]['gender']['value'] = 'M'
                elif(currValue == False): 
                    row['authors'][i]['gender']['value'] = 'F'
                
            if(firstOutput): 
                outputFile.write(',')
            else:
                firstOutput = True
            outputFile.write(str(json.dumps(row)) + '\n')

    outputFile.write(']')

def convertIMDB(tsvFile: str, outputFile: str):
    header = None
    outputFile = open(outputFile, 'w')
    i = 0
    for line in open(tsvFile, 'r'):
        if(not header): 
            header = line
            outputFile.write(line)
            continue

        i += 1
        print(f"Processing #{i}")

        line = line[:-1].split('\t')
        if(line[2] == 'True'): 
            line[2] = 'M'
        elif(line[2] == 'False'):
            line[2] = 'F' 

        outputFile.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\t{line[6]}\t{line[7]}\n")

    



# convertDBLP('../dblp_labelledGender.json', '../dblp_labelledGender_updated.json')

# convertIMDB('../name.basics_labelled.tsv', '../name.basics_labelled_MF.tsv')
