import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, prim_seq_window_size, prim_seq_vocab_size, sst_window_size, sst_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.prim_seq_vocab_size = prim_seq_vocab_size # The size of the prim_seq vocab
		self.sst_vocab_size = sst_vocab_size # The size of the sst vocab

		self.prim_seq_window_size = prim_seq_window_size # The prim_seq window size
		self.sst_window_size = sst_window_size # The sst window size
		######^^^ DO NOT CHANGE ^^^##################

		self.batch_size = 100
		self.embedding_size = 100
		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		# Define sst and prim_seq embedding layers:
		self.prim_seq_embedding = tf.Variable(tf.random.normal([self.prim_seq_vocab_size, self.embedding_size], stddev=.1))
		self.sst_embedding = tf.Variable(tf.random.normal([self.sst_vocab_size, self.embedding_size], stddev=.1))
		
		# Create positional encoder layers
		self.positional_encoder_prim_seq = transformer.Position_Encoding_Layer(self.prim_seq_window_size, self.embedding_size)
		self.positional_encoder_sst = transformer.Position_Encoding_Layer(self.sst_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, True)
	
		# Define dense layer(s)
		self.dense_1 = tf.keras.layers.Dense(self.sst_vocab_size, activation='softmax')


	@tf.function
	def call(self, encoder_input, decoder_input, force_teacher = True):
		"""
		:param encoder_input: batched ids corresponding to prim_seq sentences
		:param decoder_input: batched ids corresponding to sst sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x sst_vocab_size]
		"""
	
		# TODO:
		
		# prim_seq = seq / encoder_input
		# sst = sst3/sst8 / decoder_input
		prim_seq_embeddings = tf.nn.embedding_lookup(self.prim_seq_embedding, encoder_input)
		sst_embeddings = tf.nn.embedding_lookup(self.sst_embedding, decoder_input)

		prim_seq_position = self.positional_encoder_prim_seq.call(prim_seq_embeddings)
		sst_position = self.positional_encoder_sst.call(sst_embeddings)

		prim_seq_sum = prim_seq_embeddings + prim_seq_position
		eng_sum = sst_embeddings + sst_position
		print("shapes")
		print(eng_sum.shape)
		print(prim_seq_sum.shape)
		prim_seq_encoded = self.encoder(prim_seq_sum)

		if force_teacher:
			decoder_output = self.decoder(eng_sum, context = prim_seq_encoded)
		else:
			# TODO: how to effectively implement testing / inference without teacher forcing
			decoder_output = self.decoder(prim_seq_encoded, context = prim_seq_encoded)

		dense = self.dense_1(decoder_output)
	
		return dense

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x sst_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))

		return accuracy

	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x sst_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		loss = tf.boolean_mask(loss, mask)
		loss = tf.math.reduce_sum(loss)
		return loss

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
