import csv
import numpy as np

def opener(filename : str):
    with open(filename, newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='|')

        seq_vocab = dict() # 2
        seq_sequences = list()
        sst8_vocab = dict() # 3
        sst8_sequences = list()
        sst3_vocab = dict() # 4
        sst3_sequences = list()
        
        counter = 0
        for row in filereader:
            
            if (row[6] == 'False'): # only choose standard aa sequences
                seq = list(row[2])
                sst8 = list(row[3])
                sst3 = list(row[4])

                seq_nums = addToDict(seq_vocab, seq)
                sst8_nums = addToDict(sst8_vocab, sst8)
                sst3_nums = addToDict(sst3_vocab, sst3)

                #print(seq_nums)
                
                seq_sequences.append(seq_nums)
                sst8_sequences.append(sst8_nums)
                sst3_sequences.append(sst3_nums)
    
        #print(len(seq_sequences))
        #print(seq_sequences[0:5])
        seq_window, seq_mask = listToNumpyWindowed(seq_sequences)
        sst8_window, sst8_mask = listToNumpyWindowed(sst8_sequences)
        sst3_window, sst3_mask = listToNumpyWindowed(sst3_sequences)

        print(seq_window.shape)
        print(sst3_mask.shape)
        
        return seq_vocab, seq_window, seq_mask, sst8_vocab, sst8_window, sst8_mask, sst3_vocab, sst3_window, sst3_mask
            
def addToDict(dicter, char_list):
    for char in char_list:
        if char not in dicter:
            value = 0 if len(dicter) == 0 else max(dicter.values()) + 1
            dicter[char] = value
            
    seqs = [dicter[i] for i in char_list]
    
    return seqs

def listToNumpyWindowed(sequences, window_size = 30, padding_symbol = -1, add_start = True, start_symbol = 50, add_stop = True, stop_symbol = 99):
    windowed = list()
    masked = list()
    for seq in sequences:
        
        seqlen = len(seq)
        
        if (seqlen < window_size):
            
            seqqer = [(seq[i] if i < seqlen else padding_symbol) for i in range(0, window_size)]
            masker = [(1 if i < seqlen else 0) for i in range(0, window_size)]

            if (add_stop):
                seqqer[seqlen] = stop_symbol
                seqqer.append(padding_symbol)
                masker.append(0)

            if (add_start):
                #seqqer = [start_symbol] + [seqqer]
                seqqer.insert(0, start_symbol)
                #masker = [0] + [masker]
                masker.insert(0, 0)
            
            windowed.append(seqqer)
            masked.append(masker)
            
            #print(seqqer)
            #print(len(seqqer))
        elif (seqlen == window_size):
            seqqer = seq
            masker = [(1 if i < seqlen else 0) for i in range(0, window_size)]

            if (add_stop):
                seqqer.append(stop_symbol)
                masker.append(0)

            if (add_start):
                seqqer.insert(0, start_symbol)
                masker.insert(0, 0)


            windowed.append(seqqer)
            masked.append(masker)

        else:

            for k in range(0, int(seqlen/window_size) + 1):
                temper = seq[k * window_size : (k+ 1) * window_size]
                
                if (len(temper) > 0 and len(temper) <= window_size):
                    seqqer = [(temper[i] if i < len(temper) else padding_symbol) for i in range(0, window_size)]
                    masker = [(1 if i < len(temper) else 0) for i in range(0, window_size)]

                    if (add_stop):
                        if (len(temper) == window_size):
                            seqqer.append(stop_symbol)
                            masker.append(0)
                        else:
                            seqqer[len(temper)] = stop_symbol
                            seqqer.append(padding_symbol)
                            masker.append(0)
                        
                    if (add_start):
                        seqqer.insert(0, start_symbol)
                        masker.insert(0, 0)
                    
                    windowed.append(seqqer)
                    masked.append(masker)

    print(len(windowed))
    print(windowed[len(windowed) - 2])
    print(windowed[len(windowed) - 1])
    return np.array(windowed), np.array(masked)
    

opener("../protein_secondary_structure_data/2018-06-06-pdb-intersect-pisces.csv")
