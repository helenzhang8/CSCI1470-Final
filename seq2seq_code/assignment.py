import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from openfiles import *
from transformer_model import Transformer_Seq2Seq
import sys
import random

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
PRIMARY_WINDOW_SIZE = 30
SECONDARY_WINDOW_SIZE = 30
##########DO NOT CHANGE#####################

def train(model, train_primary, train_secondary, train_secondary_mask):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_primary: primary train data (all data for training) of shape (num_sentences, 14)
	:param train_secondary: secondary train data (all data for training) of shape (num_sentences, 15)
	:param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	input_size = train_primary.shape[0]
	indices = tf.range(start=0, limit=input_size, dtype=tf.int32)
	idx = tf.random.shuffle(indices)

	train_primary = tf.gather(train_primary, idx)
	train_secondary = tf.gather(train_secondary, idx)


	for i in range(0, input_size, model.batch_size):
		with tf.GradientTape() as tape:
			logits = model.call(train_primary[i:i + model.batch_size], train_secondary[i:i + model.batch_size])
			loss = model.loss_function(logits, train_secondary[i:i + model.batch_size], train_secondary_mask[i:i + model.batch_size])
			print(i, loss)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_primary, test_secondary, test_secondary_mask):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_primary: primary test data (all data for testing) of shape (num_sentences, 14)
	:param test_secondary: secondary test data (all data for testing) of shape (num_sentences, 15)
	:param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!

	input_size = test_primary.shape[0]

	accuracy_acc = 0
	loss_acc = 0
	total_words = 0

	for i in range(0, input_size, model.batch_size):
		probabilities = model.call(test_primary[i:i + model.batch_size], test_secondary[i:i + model.batch_size])

		words = tf.cast(tf.reduce_sum(sum(test_secondary_mask)), dtype=tf.float32)
		total_words += words

		loss = model.loss_function(probabilities, test_secondary[i:i + model.batch_size], test_secondary_mask[i:i + model.batch_size])
		accuracy = model.accuracy_function(probabilities, test_secondary[i:i + model.batch_size], test_secondary_mask[i:i + model.batch_size])

		batch_accuracy = words * accuracy
		accuracy_acc += batch_accuracy
		loss_acc += loss

	accuracy = accuracy_acc/total_words
	avg_loss = loss_acc/total_words
	perplexity = np.exp(avg_loss)
	print("PERPLEXITY", perplexity)
	print("ACCURACY", accuracy)

	return perplexity, accuracy


def main():

	print("Running preprocessing...")
	# train_primary, test_primary, train_secondary, test_secondary, secondary_vocab, primary_vocab, secondary_padding_index = get_data('../protein_secondary_structure_data/2018-06-06-pdb-intersect-pisces.csv')
	#train_primary1, test_primary1, train_secondary1, test_secondary1, secondary_vocab1, primary_vocab1, secondary_padding_index1 = get_data('../protein_secondary_structure_data/2018-06-06-ss.cleaned.csv')
	seq_vocab, seq_window, seq_mask, sst8_vocab, sst8_window, sst8_mask, sst3_vocab, sst3_window, sst3_mask = opener("../protein_secondary_structure_data/2018-06-06-pdb-intersect-pisces.csv", PRIMARY_WINDOW_SIZE)
	print("Preprocessing complete.")

	model_args = (PRIMARY_WINDOW_SIZE, len(seq_vocab), SECONDARY_WINDOW_SIZE, len(sst8_vocab))
	train_size = 3000
	train_primary = seq_window[:train_size]
	train_secondary = sst8_window[:train_size]
	train_secondary_mask = sst8_mask[:train_size]
	test_primary = seq_window[train_size:]
	test_secondary = sst8_window[train_size:]
	test_secondary_mask = sst8_mask[train_size:]
	padding_index = -1

	# print("TRAIN_PRIMARY: ", train_primary.shape)
	# print(train_secondary.shape)
	# print(test_primary.shape)
	# print(test_secondary.shape)

	model = Transformer_Seq2Seq(*model_args)

	train(model, train_primary, train_secondary, train_secondary_mask)
	test(model, test_primary, test_secondary, test_secondary_mask)


if __name__ == '__main__':
	main()
