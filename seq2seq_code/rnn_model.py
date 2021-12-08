import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, primary_window_size, primary_vocab_size, secondary_window_size, secondary_vocab_size, embedding_size, learning_rate):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.primary_vocab_size = primary_vocab_size # The size of the primary vocab
		self.secondary_vocab_size = secondary_vocab_size # The size of the secondary vocab

		self.primary_window_size = primary_window_size # The primary window size
		self.secondary_window_size = secondary_window_size # The secondary window size

		self.embedding_size = embedding_size
		self.learning_rate = learning_rate
		######^^^ DO NOT CHANGE ^^^##################

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
	
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.encoder = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
		self.decoder = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
		self.dense_1 = tf.keras.layers.Dense(self.secondary_vocab_size, activation='softmax')
		# self.dense_2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
		#
		self.encoderE = tf.Variable(tf.random.normal([self.primary_vocab_size, self.embedding_size], stddev=.1))
		self.decoderE = tf.Variable(tf.random.normal([self.secondary_vocab_size, self.embedding_size], stddev=.1))

	@tf.function
	def call(self, encoder_input, decoder_input, force_teacher=False):
		"""
		:param encoder_input: batched ids corresponding to primary sentences
		:param decoder_input: batched ids corresponding to secondary sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x secondary_vocab_size]
		"""

		#1) Pass your primary sentence embeddings to your encoder
		encoder_embedding = tf.nn.embedding_lookup(self.encoderE, encoder_input)
		encoder_output, encoder_initial1, encoder_initial2 = self.encoder(encoder_embedding, initial_state=None)

		#2) Pass your secondary sentence embeddings, and final state of your encoder, to your decoder
		decoder_embedding = tf.nn.embedding_lookup(self.decoderE, decoder_input)
		decoder_output, decoder_initial1, decoder_initial2 = self.decoder(decoder_embedding, initial_state=(encoder_initial1, encoder_initial2))

		#3) Apply dense layer(s) to the decoder out to generate probabilities
		logits = self.dense_1(decoder_output)

		return logits

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x secondary_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy

	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass. 
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x secondary_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		loss = tf.boolean_mask(loss, mask)
		loss = tf.math.reduce_sum(loss)
		return loss

