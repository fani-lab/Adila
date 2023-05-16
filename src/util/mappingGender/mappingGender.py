import pickle as pkl
import pandas as pd


class MappingGender:
    def __init__(self, opeNTF_output_dir: str) -> None:
        # Converts the identifier in the dataset to opeNTF index
        self.memberId_2_i = {}
        self.df = None

        with open(opeNTF_output_dir, "rb") as f:
            self.opeNTF_out = pkl.load(f)
    
    # Generates the Mapping for Member ID to opeNTF index
    def createMemberID_2_i_IMDB(self):
        for key in self.opeNTF_out['c2i'].keys():
            new_key = key[:key.index('.')]
            new_key = "nm" + ("0" * (7-len(new_key))) + new_key
            self.memberId_2_i[new_key] = self.opeNTF_out['c2i'][key]
    
    # Generates the dataframe mapping for IMDB dataset
    def findGenderValues_IMDB(self, imdb_titleBasics_dir : str):
        f = open(imdb_titleBasics_dir, 'r')
        f.readline()
        line = f.readline()

        # Variables to form dataframe:
        indexes = []

        keysNotFound = set(self.opeNTF_out['i2c'].keys())

        data = {
            "gender" : [],
            "probability" : []
        }

        while(line != ''):
            lineArr = line.split('\t')
            print(lineArr[0])
            if(lineArr[0] in self.memberId_2_i): 
                indexes.append(self.memberId_2_i[lineArr[0]])
                keysNotFound.remove(self.memberId_2_i[lineArr[0]])
                data['gender'].append(bool(lineArr[2]))
                data['probability'].append(lineArr[3])
                
            line = f.readline()    

        # Find Keys that were not found in the file:
        for key in list(keysNotFound):
            indexes.append(key)
            data['gender'].append(None)
            data['probability'].append(None)


        self.df = pd.DataFrame(data, indexes)

        print(self.df.head(100))
    
    def exportResults_toPickle(self, directory):
        self.df.to_pickle(path=directory)
    
    def exportResults_toCSV(self, directory):
        self.df.to_csv(directory)

    def importResults(self, directory):
        self.df = pd.read_pickle(directory)


imdbMapGender = MappingGender('data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3/indexes.pkl')

imdbMapGender.createMemberID_2_i_IMDB()

imdbMapGender.findGenderValues_IMDB('../name.basics_labelled.tsv')

imdbMapGender.exportResults_toCSV('data/preprocessed/imdb/i2gender.csv')

imdbMapGender.exportResults_toPickle('data/preprocessed/imdb/i2gender.pkl')



