import csv
import numpy as np

def opener(filename : str):
    with open(filename, newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='|')

        seq_language = dict() # 2
        seq_sequences = list()
        sst8_language = dict() # 3
        sst8_sequences = list()
        sst3_language = dict() # 4
        sst3_sequences = list()
        
        counter = 0
        for row in filereader:
            
            if (row[6] == 'False'): # only choose standard aa sequences
                seq = list(row[2])
                sst8 = list(row[3])
                sst3 = list(row[4])

                seq_nums = addToDict(seq_language, seq)
                sst8_nums = addToDict(sst8_language, sst8)
                sst3_nums = addToDict(sst3_language, sst3)
                
                seq_sequences.append(seq_nums)
                sst8_sequences.append(sst8_nums)
                sst3_sequences.append(sst3_nums)
    
        print(len(seq_sequences))

        return seq_language, seq_sequences, sst8_language, sst8_sequences, sst3_language, sst3_sequences
            
def addToDict(dicter, char_list):
    for char in char_list:
        if char not in dicter:
            value = 0 if len(dicter) == 0 else max(dicter.values())
            dicter[char] = value
    return [dicter[i] for i in char_list]

opener("../protein_secondary_structure_data/2018-06-06-pdb-intersect-pisces.csv")
