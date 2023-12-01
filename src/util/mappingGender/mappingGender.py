import pickle as pkl
import pandas as pd
import json
import multiprocessing as mp


class MappingGender:
    def __init__(self, opeNTF_output_dir: str) -> None:
        # Converts the identifier in the dataset to opeNTF index
        self.memberId_2_i = {}
        self.df = None

        if(opeNTF_output_dir):
            with open(opeNTF_output_dir, "rb") as f:
                self.opeNTF_out = pkl.load(f)
        
    
    # Generates the Mapping for Member ID to opeNTF index
    def createMemberID_2_i_IMDB(self):
        for key in self.opeNTF_out['c2i'].keys():
            new_key = key[:key.index('.')]
            new_key = "nm" + ("0" * (7-len(new_key))) + new_key
            self.memberId_2_i[new_key] = self.opeNTF_out['c2i'][key]
    
    # Generates the Mapping for Member ID to opeNTF index
    def createMemberID_2_i_DBLP(self):
        for key in self.opeNTF_out['c2i'].keys():
            new_key = int(key[:key.index('_')])
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
            "rawIndex": [],
            "gender" : [],
            "probability" : []
        }

    
        while(line != ''):
            lineArr = line.split('\t')
            print(lineArr[0])
            if(lineArr[0] in self.memberId_2_i): 
                indexes.append(self.memberId_2_i[lineArr[0]])
                keysNotFound.remove(self.memberId_2_i[lineArr[0]])
                data['rawIndex'].append(lineArr[0])
                data['gender'].append(lineArr[2])
                data['probability'].append(lineArr[3])
                
            line = f.readline()    

        # Find Keys that were not found in the file:
        for key in list(keysNotFound):
            indexes.append(key)
            data['rawIndex'].append(None)
            data['gender'].append(None)
            data['probability'].append(None)


        self.df = pd.DataFrame(data, indexes)

    
    def findGenderResults_DBLP(self, dblp_json_dir :str):
        f = open(dblp_json_dir, 'r')
        f.readline()
        line = f.readline()
        
        keysNotFound = set(self.opeNTF_out['i2c'].keys())
        data = {}
        i = 0
        while(line != ']'):
            print(f"Processing {i}...")
            i += 1
            # if(i > 100000): break
            if(line[0] == ','): line = line[1:]
            row = json.loads(line)

            for author in row["authors"]:
                if(author["id"] not in self.memberId_2_i): continue
                if(author["id"] not in data):
                    if(self.memberId_2_i[author["id"]] in keysNotFound):
                        keysNotFound.remove(self.memberId_2_i[author["id"]])
                    data[self.memberId_2_i[author["id"]]] = [author["id"], author["gender"]["value"], author["gender"]["probability"]]

            line = f.readline()    

        for key in keysNotFound:
            data[key] = [None, None, None]
        
        self.df = pd.DataFrame.from_dict(data, orient="index", columns=["rawIndex", "gender", "probability"])

    # Generates the dataframe mapping for IMDB dataset
    def findGenderValues_IMDB_v2(self, imdb_titleBasics_dir : str):
        f = open(imdb_titleBasics_dir, 'r')
        f.readline()
        line = f.readline()

        # Variables to form dataframe:
        indexes = []

        keysNotFound = set(self.opeNTF_out['i2c'].keys())

        data = {
            "gender" : []
        }

    
        while(line != ''):
            lineArr = line.split('\t')
            print(lineArr[0])
            if(lineArr[0] in self.memberId_2_i): 
                indexes.append(self.memberId_2_i[lineArr[0]])
                keysNotFound.remove(self.memberId_2_i[lineArr[0]])
                if(lineArr[2] == 'M'): 
                    data['gender'].append(True)
                else:
                    data['gender'].append(False)
                
            line = f.readline()    

        # Find Keys that were not found in the file:
        for key in list(keysNotFound):
            indexes.append(key)
            data['gender'].append(True)

        self.df = pd.DataFrame(data, indexes)

    
    def findGenderResults_DBLP_v2(self, dblp_json_dir :str):
        f = open(dblp_json_dir, 'r')
        f.readline()
        line = f.readline()
        
        keysNotFound = set(self.opeNTF_out['i2c'].keys())
        data = {}
        i = 0
        while(line != ']'):
            print(f"Processing {i}...")
            i += 1
            # if(i > 100000): break
            if(line[0] == ','): line = line[1:]
            row = json.loads(line)

            for author in row["authors"]:
                if(author["id"] not in self.memberId_2_i): continue
                if(author["id"] not in data):
                    if(self.memberId_2_i[author["id"]] in keysNotFound):
                        keysNotFound.remove(self.memberId_2_i[author["id"]])
                    if(author["gender"]["value"] == 'M'):
                        data[self.memberId_2_i[author["id"]]] = [True]
                    else:
                        data[self.memberId_2_i[author["id"]]] = [False]

            line = f.readline()    

        for key in keysNotFound:
            data[key] = [True]
        
        self.df = pd.DataFrame.from_dict(data, orient="index", columns=["gender"])

    def generate_mapping_uspt(self, teams_pkl: str, indexes_pkl: str):
        """
        Generates gender.csv file to map OpeNTF ID to gender value
        True: Male, False: Female, Null: null
        Args:
            teams_pkl: location of teams.pkl file for uspt
            indexes_pkl: location of indexes.pkl file for uspt
        Return:
            None
        """
        mappings = {}

        with open(teams_pkl, "rb") as f_1:
            with open(indexes_pkl, "rb") as f_2:
                teams_pkl = pkl.load(f_1)
                indexes_pkl = pkl.load(f_2)
                c2i = indexes_pkl['c2i']

                for patent in teams_pkl:
                    for member in patent.members:
                        ind = c2i[member.id + "_" + member.name]
                        if(ind not in mappings):
                            mappings[ind] = member.gender
                                
                self.df = pd.DataFrame.from_dict(mappings, orient="index", columns=["gender"])



    def exportResults_toPickle(self, directory):
        self.df.to_pickle(path=directory)
    
    def exportResults_toCSV(self, directory):
        self.df.to_csv(directory)

    def importResults(self, directory):
        self.df = pd.read_pickle(directory)





# imdbMapGender = MappingGender('data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3/indexes.pkl')

# imdbMapGender.createMemberID_2_i_IMDB()

# imdbMapGender.findGenderValues_IMDB_v2('../name.basics_labelled_MF.tsv')

# imdbMapGender.exportResults_toCSV('data/preprocessed/imdb/i2gender.csv')

# imdbMapGender.exportResults_toPickle('data/preprocessed/imdb/i2gender.pkl')


# dblpMapGender = MappingGender('data/preprocessed/dblp/dblp.v12.json/indexes.pkl')

# dblpMapGender.createMemberID_2_i_DBLP()

# dblpMapGender.findGenderResults_DBLP_v2('../dblp_labelledGender_updated.json')

# dblpMapGender.exportResults_toCSV('data/preprocessed/dblp/i2gender.csv')

# dblpMapGender.exportResults_toPickle('data/preprocessed/dblp/i2gender.pkl')



# USPT:

uspt_map_gender = MappingGender(None)

uspt_map_gender.generate_mapping_uspt(teams_pkl="data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3/teams.pkl",
                                      indexes_pkl="data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3/indexes.pkl")


uspt_map_gender.exportResults_toCSV("data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3/gender.csv")
uspt_map_gender.exportResults_toPickle("data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3/gender.pkl")
        