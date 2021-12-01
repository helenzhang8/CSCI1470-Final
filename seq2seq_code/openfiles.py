import csv
import numpy as np


def opener(filename : str, window_size):
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
            counter += 1
            
            if row[6] == 'False': # only choose standard aa sequences
                seq = list(row[2])
                sst8 = list(row[3])
                sst3 = list(row[4])

                seq_nums = addToDict(seq_vocab, seq)
                sst8_nums = addToDict(sst8_vocab, sst8)
                sst3_nums = addToDict(sst3_vocab, sst3)
                
                seq_sequences.append(seq_nums)
                sst8_sequences.append(sst8_nums)
                sst3_sequences.append(sst3_nums)

        seq_window, seq_mask = listToNumpyWindowed(seq_sequences, window_size, add_start=False, vocab=seq_vocab)
        sst8_window, sst8_mask = listToNumpyWindowed(sst8_sequences, window_size, vocab=sst8_vocab)
        sst3_window, sst3_mask = listToNumpyWindowed(sst3_sequences, window_size, vocab=sst3_vocab)
        
        return seq_vocab, seq_window, seq_mask, sst8_vocab, sst8_window, sst8_mask, sst3_vocab, sst3_window, sst3_mask


def addToDict(dicter, char_list):
    for char in char_list:
        if char not in dicter:
            value = 1 if len(dicter) == 0 else max(dicter.values()) + 1  # reserve 0 for padding
            dicter[char] = value
            
    seqs = [dicter[i] for i in char_list]
    
    return seqs


def listToNumpyWindowed(sequences, window_size=30, padding_symbol=0, add_start=True, add_stop=True, vocab=None):
    windowed = list()
    masked = list()
    if vocab:
        if add_start:
            vocab["START"] = max(vocab.values()) + 1
            start_symbol = vocab["START"]
        if add_stop:
            vocab["STOP"] = max(vocab.values()) + 1
            stop_symbol = vocab["STOP"]
        vocab["PADDING"] = 0

    if add_stop:
        window_size -= 1

    print("SEQ LENGTH: ", len(sequences))
    for seq in sequences:

        seqlen = len(seq)
        
        if seqlen < window_size:
            
            seqqer = [(seq[i] if i < seqlen else padding_symbol) for i in range(0, window_size)]
            masker = [(1 if i < seqlen else 0) for i in range(0, window_size)]

            if add_stop:
                seqqer[seqlen] = stop_symbol
                seqqer.append(padding_symbol)
                masker.append(0)

            if add_start:
                seqqer.insert(0, start_symbol)
                masker.insert(0, 0)

            windowed.append(seqqer)
            masked.append(masker)

        elif seqlen == window_size:
            seqqer = seq
            masker = [(1 if i < seqlen else 0) for i in range(0, window_size)]

            if add_stop:
                seqqer.append(stop_symbol)
                masker.append(0)

            if add_start:
                seqqer.insert(0, start_symbol)
                masker.insert(0, 0)

            windowed.append(seqqer)
            masked.append(masker)

        else:
            for k in range(0, int(seqlen/window_size) + 1):
                temper = seq[k * window_size: (k + 1) * window_size]
                
                if 0 < len(temper) <= window_size:
                    seqqer = [(temper[i] if i < len(temper) else padding_symbol) for i in range(0, window_size)]
                    masker = [(1 if i < len(temper) else 0) for i in range(0, window_size)]

                    if add_stop:
                        if len(temper) == window_size:
                            seqqer.append(stop_symbol)
                            masker.append(0)
                        else:
                            seqqer[len(temper)] = stop_symbol
                            seqqer.append(padding_symbol)
                            masker.append(0)
                        
                    if add_start:
                        seqqer.insert(0, start_symbol)
                        masker.insert(0, 0)
                    
                    windowed.append(seqqer)
                    masked.append(masker)

    return np.array(windowed), np.array(masked)