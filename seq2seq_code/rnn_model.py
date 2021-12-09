import numpy as np
import tensorflow as tf
 
class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size, embedding_size, learning_rate):
        ###### DO NOT CHANGE ##############
        super(RNN_Seq2Seq, self).__init__()
        self.french_vocab_size = french_vocab_size # The size of the french vocab
        self.english_vocab_size = english_vocab_size # The size of the english vocab
 
        self.french_window_size = french_window_size # The french window size
        self.english_window_size = english_window_size # The english window size
 
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        ######^^^ DO NOT CHANGE ^^^##################
 
 
        # TODO:
        # 1) Define any hyperparameters
 
        # Define batch size and optimizer/learning rate
 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.encoder = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')
        # self.dense_2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        #
        self.encoderE = tf.Variable(tf.random.normal([self.french_vocab_size, self.embedding_size], stddev=.1))
        self.decoderE = tf.Variable(tf.random.normal([self.english_vocab_size, self.embedding_size], stddev=.1))
 
    @tf.function
    def call(self, encoder_input, decoder_input, force_teacher=False):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
 
        #1) Pass your french sentence embeddings to your encoder
        encoder_embedding = tf.nn.embedding_lookup(self.encoderE, encoder_input)
        encoder_output, encoder_initial1, encoder_initial2 = self.encoder(encoder_embedding, initial_state=None)
 
        #2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
        if force_teacher:
            decoder_embedding = tf.nn.embedding_lookup(self.decoderE, decoder_input)
            decoder_output, decoder_initial1, decoder_initial2 = self.decoder(decoder_embedding, initial_state=(encoder_initial1, encoder_initial2))
        else:
            decoder_output, decoder_initial1, decoder_initial2 = self.decoder(encoder_output, initial_state=(encoder_initial1, encoder_initial2))
 
        #3) Apply dense layer(s) to the decoder out to generate probabilities
        logits = self.dense_1(decoder_output)
 
        return logits
 
    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE
 
        Computes the batch accuracy
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
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
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """
 
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
        loss = tf.boolean_mask(loss, mask)
        loss = tf.math.reduce_sum(loss)
        return loss
 