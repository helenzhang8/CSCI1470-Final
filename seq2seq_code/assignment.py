import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys
import random


def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	input_size = train_french.shape[0]

	for i in range(0, input_size, model.batch_size):
		with tf.GradientTape() as tape:
			logits = model.call(train_french[i:i + model.batch_size], train_english[i:i + model.batch_size, :-1])
			loss = model.loss_function(logits, train_english[i:i + model.batch_size, 1:], np.where(train_english[i:i + model.batch_size, 1:] == eng_padding_index, 0, 1))
			print(i, loss)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!

	input_size = test_french.shape[0]

	accuracy_acc = 0
	loss_acc = 0
	total_words = 0

	for i in range(0, input_size, model.batch_size):
		probabilities = model.call(test_french[i:i + model.batch_size], test_english[i:i + model.batch_size, :-1])

		mask = (test_english[i:i + model.batch_size, 1:] != eng_padding_index)
		words = tf.cast(tf.reduce_sum(sum(mask)), dtype=tf.float32)
		total_words += words

		loss = model.loss_function(probabilities, test_english[i:i + model.batch_size, 1:], np.where(test_english[i:i + model.batch_size, 1:] == eng_padding_index, 0, 1))
		accuracy = model.accuracy_function(probabilities, test_english[i:i + model.batch_size, 1:], np.where(test_english[i:i + model.batch_size, 1:] == eng_padding_index, 0, 1))

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
	train_primary, test_primary, train_secondary, test_secondary, secondary_vocab, primary_vocab, secondary_padding_index = get_data('../protein_secondary_structure_data/2018-06-06-pdb-intersect-pisces.csv')
	print("Preprocessing complete.")

	model_args = (PRIMARY_WINDOW_SIZE, len(primary_vocab), SECONDARY_WINDOW_SIZE, len(secondary_vocab))

	model = Transformer_Seq2Seq(*model_args)

	train(model, train_primary, train_secondary, secondary_padding_index)
	#test(model, test_primary, test_secondary, secondary_padding_index)


if __name__ == '__main__':
	main()
