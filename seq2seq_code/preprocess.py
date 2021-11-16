import numpy as np
import tensorflow as tf

from attenvis import AttentionVis
av = AttentionVis()

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
PRIMARY_WINDOW_SIZE = 10
SECONDARY_WINDOW_SIZE = 10
##########DO NOT CHANGE#####################


def pad_corpus(primary, secondary):
	"""
	DO NOT CHANGE:

	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	"""
	PRIMARY_padded_sentences = []
	PRIMARY_sentence_lengths = []
	for line in primary:
		line = list(line)
		padded_PRIMARY = line[:PRIMARY_WINDOW_SIZE-1]
		padded_PRIMARY += [STOP_TOKEN] + [PAD_TOKEN] * (PRIMARY_WINDOW_SIZE - len(padded_PRIMARY)-1)
		PRIMARY_padded_sentences.append(padded_PRIMARY)

	SECONDARY_padded_sentences = []
	SECONDARY_sentence_lengths = []
	for line in secondary:
		line = list(line)
		padded_SECONDARY = line[:SECONDARY_WINDOW_SIZE-1]
		padded_SECONDARY = [START_TOKEN] + padded_SECONDARY + [STOP_TOKEN] + [PAD_TOKEN] * (SECONDARY_WINDOW_SIZE - len(padded_SECONDARY)-1)
		SECONDARY_padded_sentences.append(padded_SECONDARY)

	return PRIMARY_padded_sentences, SECONDARY_padded_sentences


def build_vocab(sentences):
	"""
	DO NOT CHANGE

	Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  	"""

	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab, vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

	Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indices in the corresponding sentences
	"""
	converted = np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])
	return converted


def read_data(file_name):
	"""
	DO NOT CHANGE

	Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
	"""
	primary_sequences = []
	secondary_sequences = []
	#with open(file_name, 'rt', encoding='latin') as data_file:
	data_file = open(file_name, 'rt', encoding='latin')
	data_file.readline()
	for line in data_file:
		split = line.split(',')
		primary_sequences.append(split[2])
		secondary_sequences.append(split[4]) #Q3
	return primary_sequences, secondary_sequences


@av.get_data_func
def get_data(file_name):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	
	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	english vocab (Dict containing word->index mapping),
	french vocab (Dict containing word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	#1) Read English and French Data for training and testing (see read_data)
	read_file = read_data(file_name)
	primary_sequence = read_file[0]
	secondary_sequence = read_file[1]

	pad = pad_corpus(primary_sequence, secondary_sequence)

	primary_vocab = build_vocab(pad[0])[0]

	secondary_vocab_out = build_vocab(pad[1])
	secondary_vocab = secondary_vocab_out[0]
	secondary_padding_index = secondary_vocab_out[1]

	train_primary = convert_to_id(primary_vocab, pad[0][:7000])
	test_primary = convert_to_id(primary_vocab, pad[0][7000:])

	train_secondary = convert_to_id(secondary_vocab, pad[1][:7000])
	test_secondary = convert_to_id(secondary_vocab, pad[1][7000:])

	return train_primary, test_primary, train_secondary, test_secondary, primary_vocab, secondary_vocab, secondary_padding_index
	