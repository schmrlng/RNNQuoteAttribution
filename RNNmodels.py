import sys
import os
import time

import numpy as np
import tensorflow as tf
import gensim

from tensorflow.python.ops.rnn_cell import RNNCell, BasicRNNCell, GRUCell, DropoutWrapper
from q2_initialization import xavier_weight_init

from collections import defaultdict
import matplotlib.pyplot as plt

from utils import Vocab, Speakers, load_chapter_split, data_iterator, calculate_confusion, print_confusion
from tensorflow_birnn import bidirectional_dynamic_rnn

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    # wordvecpath = "glove.840B.300d.filtered.txt"
    wordvecpath = "glove.6B.100d.filtered.txt"
    
    datapath = "futurama/futurama.txt"
    datasplitpath = "futurama/futurama_split.txt"
    speaker_count = 8
    # datapath = "prideprejudice/prideprejudice.txt"
    # datasplitpath = "prideprejudice/prideprejudice_split.txt"
    # speaker_count = 4

    max_line_length = 500  # for chopping

    embed_size = int(wordvecpath.split(".")[2][:-1])
    # batch_size = 120
    batch_size = "chapter"
    early_stopping = 2
    max_epochs = 60
    dropout = 0.9
    lr = 0.002
    l2 = 0.0001
    anneal_threshold = .995
    anneal_by = 1.5
    weight_loss = True
    bidirectional_sentences = True
    bidirectional_conversations = True

class WhoseLineModel(object):

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_common_model_vars()
        
    def load_data(self, debug=False):
        self.wordvecs = gensim.models.Word2Vec.load_word2vec_format(self.config.wordvecpath, binary=False)
        self.vocab = Vocab()
        self.vocab.construct(self.wordvecs.index2word)
        self.embedding_matrix = np.vstack([self.wordvecs[self.vocab.index_to_word[i]] for i in range(len(self.vocab))])
        # next line is "unk" surgery cf. https://groups.google.com/forum/#!searchin/globalvectors/unknown/globalvectors/9w8ZADXJclA/X6f0FgxUnMgJ
        self.embedding_matrix[0,:] = np.mean(self.embedding_matrix, axis=0)

        chapter_split = load_chapter_split(self.config.datasplitpath)
        self.speakers = Speakers()
        for line in open(self.config.datapath):
            ch, speaker, line = line.split("\t")
            if chapter_split[ch] == 0:
                self.speakers.add_speaker(speaker)
        self.speakers.prune(self.config.speaker_count-1)  # -1 for OTHER

        self.train_data = []
        self.dev_data = []
        self.test_data = []
        oldch = None
        for ln in open(self.config.datapath):
            ch, speaker, line = ln.split("\t")
            encoded_line = (np.array([self.vocab.encode(word) for word in line.split()], dtype=np.int32),
                            self.speakers.encode(speaker))
            if chapter_split[ch] == 0:
                dataset = self.train_data
            elif chapter_split[ch] == 1:
                dataset = self.dev_data
            else:
                dataset = self.test_data
            if self.config.batch_size == "chapter":
                if ch == oldch:
                    dataset[-1].append(encoded_line)
                else:
                    dataset.append([encoded_line])
            else:
                dataset.append(encoded_line)
            oldch = ch
    
    def add_common_model_vars(self):
        with tf.variable_scope("word_vectors"):
            self.tf_embedding_matrix = tf.constant(self.embedding_matrix, name="embedding")

class SumRNNCell(RNNCell):
    """The (even most-er) most basic RNN cell."""

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """(Most-er) most basic RNN: output = new_state = input + state."""
        output = inputs + state
        return output, output
    
class WhoseLineSoftmaxModel(WhoseLineModel):
    
    def add_placeholders(self):
        self.lines_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.line_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.loss_weights_placeholder = tf.placeholder(tf.float32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.l2_placeholder = tf.placeholder(tf.float32)
        
    def create_feed_dict(self, lines_batch, line_length_batch, labels_batch, dropout, l2, loss_weights_batch = None):
        if loss_weights_batch == None:
            loss_weights_batch = np.ones_like(labels_batch, np.float32)
        feed_dict = {self.lines_placeholder: lines_batch,
                     self.line_length_placeholder: line_length_batch,
                     self.labels_placeholder: labels_batch,
                     self.dropout_placeholder: dropout,
                     self.l2_placeholder: l2,
                     self.loss_weights_placeholder: loss_weights_batch}
        return feed_dict
        
    def add_model(self):
        embed_size = self.config.embed_size
        num_speakers = self.config.speaker_count
        self.embedded_lines = tf.gather(self.tf_embedding_matrix, self.lines_placeholder)
        sentence_summary_size = self.add_sentence_summaries(embed_size)
        conversation_state_size = self.add_conversational_context(sentence_summary_size)
        with tf.variable_scope("linear_softmax"):
            W = tf.get_variable("weights", (conversation_state_size, num_speakers), initializer=xavier_weight_init())
            b = tf.get_variable("biases", (num_speakers,))
        return tf.nn.dropout(tf.matmul(self.conversation_state, W) + b, self.dropout_placeholder)  # logits

    def add_sentence_summaries(self, wordvector_embed_size):
        # simple average of the wordvectors in the sentence
        sumrnncell = SumRNNCell(wordvector_embed_size)
        _, state = tf.nn.dynamic_rnn(sumrnncell, self.embedded_lines, self.line_length_placeholder, dtype = tf.float32)
        self.sentence_summaries = tf.div(state, tf.to_float(tf.reshape(self.line_length_placeholder, (-1, 1))))
        return wordvector_embed_size

    def add_conversational_context(self, sentence_summary_size):
        # no context is taken into account; classification is based purely on sentence content
        with tf.variable_scope("hidden_layer"):
            W = tf.get_variable("weights", (sentence_summary_size, sentence_summary_size), initializer=xavier_weight_init())
            b = tf.get_variable("biases", (sentence_summary_size,))
        self.conversation_state = tf.tanh(tf.nn.dropout(tf.matmul(self.sentence_summaries, W) + b, self.dropout_placeholder))
        return sentence_summary_size

    def add_loss_op(self, y):
        loss = tf.reduce_mean(tf.mul(tf.nn.sparse_softmax_cross_entropy_with_logits(y, self.labels_placeholder), self.loss_weights_placeholder))
        weight_matrices = [v for v in tf.all_variables() if "weights" in v.name or "Matrix" in v.name]
        print [wmat.name for wmat in weight_matrices]
        for wmat in weight_matrices:
            loss = loss + self.l2_placeholder*tf.nn.l2_loss(wmat)
        return loss
    
    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(self.config.lr)
        train_op = opt.minimize(loss)
        return train_op
    
    def __init__(self, config):
        super(WhoseLineSoftmaxModel, self).__init__(config)
        self.add_placeholders()
        y = self.add_model()
        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        self.one_hot_predictions = tf.argmax(self.predictions, 1)
        self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(self.labels_placeholder, tf.to_int32(self.one_hot_predictions)), 'int32'))
        self.train_op = self.add_training_op(self.loss)
    
    def run_epoch(self, session, input_data, shuffle=False, verbose=True):
        dp = self.config.dropout
        l2 = self.config.l2
        # We're interested in keeping track of the loss and accuracy during training
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        if self.config.batch_size == "chapter":
            total_steps = len(input_data)
        else:
            total_steps = len(input_data) / self.config.batch_size
        data = data_iterator(input_data, batch_size=self.config.batch_size, chop_limit=self.config.max_line_length, shuffle=shuffle)
        for step, (lines, line_lengths, labels) in enumerate(data):
            if self.config.weight_loss:
                # f = np.log
                f = lambda x: np.sqrt(x)
                normalization_factor = np.mean([1./f(ct) for ct in self.speakers.speaker_freq.values()])
                self.index_to_weight = {k:1./(normalization_factor*f(self.speakers.speaker_freq[v])) for k,v in self.speakers.index_to_speaker.items()}
                feed = self.create_feed_dict(lines, line_lengths, labels, dp, l2, [self.index_to_weight[l] for l in labels])
            else:
                self.index_to_weight = {k:1. for k in range(len(self.speakers))}
                feed = self.create_feed_dict(lines, line_lengths, labels, dp, l2)
            loss, total_correct, _ = session.run([self.loss, self.correct_predictions, self.train_op], feed_dict=feed)
            total_processed_examples += len(labels)
            total_correct_examples += total_correct
            total_loss.append(loss)
            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)
    
    def predict(self, session, input_data, disallow_other=False):
        dp = 1.
        l2 = 0.
        losses = []
        results = []
        data = data_iterator(input_data, batch_size=self.config.batch_size, chop_limit=self.config.max_line_length)
        for step, (lines, line_lengths, labels) in enumerate(data):
            feed = self.create_feed_dict(lines, line_lengths, labels, dp, l2, [self.index_to_weight[l] for l in labels])
            loss, preds, predicted_indices = session.run([self.loss, self.predictions, self.one_hot_predictions], feed_dict=feed)
            if disallow_other:
                preds[:,self.speakers.speaker_to_index["OTHER"]] = 0.
                predicted_indices = preds.argmax(axis=1)
            losses.append(loss)
            results.extend(predicted_indices)
        return np.mean(losses), results

class WhoseLine_RNN_NoContext_Model(WhoseLineSoftmaxModel):

    def add_sentence_summaries(self, wordvector_embed_size):
        if self.config.bidirectional_sentences:
            forwardcell = DropoutWrapper(GRUCell(wordvector_embed_size), self.dropout_placeholder, self.dropout_placeholder)
            backwardcell = DropoutWrapper(GRUCell(wordvector_embed_size), self.dropout_placeholder, self.dropout_placeholder)
            _, statefw, statebw = bidirectional_dynamic_rnn(forwardcell, backwardcell,
                                                            self.embedded_lines, self.line_length_placeholder,
                                                            dtype = tf.float32, scope = "LineRNN")
            self.sentence_summaries = tf.concat(1, [statefw, statebw])
            return 2*wordvector_embed_size
        else:
            rnncell = DropoutWrapper(GRUCell(wordvector_embed_size), self.dropout_placeholder, self.dropout_placeholder)
            _, self.sentence_summaries = tf.nn.dynamic_rnn(rnncell,
                                                           self.embedded_lines, self.line_length_placeholder,
                                                           dtype = tf.float32, scope = "LineRNN")
            return wordvector_embed_size

class WhoseLine_MeanWV_RNN_Model(WhoseLineSoftmaxModel):

    def add_conversational_context(self, sentence_summary_size):
        line_vectors_as_timesteps = tf.expand_dims(self.sentence_summaries, 0)
        if self.config.bidirectional_conversations:
            forwardcell = DropoutWrapper(GRUCell(sentence_summary_size), self.dropout_placeholder, self.dropout_placeholder)
            backwardcell = DropoutWrapper(GRUCell(sentence_summary_size), self.dropout_placeholder, self.dropout_placeholder)
            outputs, sf, sb = bidirectional_dynamic_rnn(forwardcell, backwardcell,
                                                        line_vectors_as_timesteps,
                                                        tf.slice(tf.shape(line_vectors_as_timesteps),[1],[1]),  # what the fucking fuck
                                                        dtype = tf.float32, scope = "ChapterRNN")
            self.conversation_state = tf.squeeze(outputs)
            return 2*sentence_summary_size
        else:
            rnncell = DropoutWrapper(GRUCell(sentence_summary_size), self.dropout_placeholder, self.dropout_placeholder)
            outputs, state = tf.nn.dynamic_rnn(rnncell,
                                               line_vectors_as_timesteps,
                                               tf.slice(tf.shape(line_vectors_as_timesteps),[1],[1]),  # what the fucking fuck
                                               dtype = tf.float32, scope = "ChapterRNN")
            self.conversation_state = tf.squeeze(outputs)
            return sentence_summary_size

class WhoseLine_RNN_RNN_Model(WhoseLine_RNN_NoContext_Model, WhoseLine_MeanWV_RNN_Model):       
    pass


def test(sentencernn, contextrnn, sentencebi, contextbi):
    config = Config()
    config.bidirectional_sentences = sentencebi
    config.bidirectional_conversations = contextbi
    with tf.Graph().as_default():
        if not sentencernn and not contextrnn:
            model = WhoseLineSoftmaxModel(config)
        elif sentencernn and not contextrnn:
            model = WhoseLine_RNN_NoContext_Model(config)
        elif not sentencernn and contextrnn:
            model = WhoseLine_MeanWV_RNN_Model(config)
        else:
            model = WhoseLine_RNN_RNN_Model(config)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            prev_train_loss = float('inf')
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            for epoch in xrange(config.max_epochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                ###
                train_loss, train_acc = model.run_epoch(session, model.train_data)
                val_loss, predictions = model.predict(session, model.dev_data)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Validation loss: {}'.format(val_loss)

                #lr annealing
                if train_loss > prev_train_loss*model.config.anneal_threshold:
                    model.config.lr /= model.config.anneal_by
                    print 'annealed lr to %f'%model.config.lr
                prev_train_loss = train_loss

                # save if model has improved on val
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    saver.save(session, './weights/whose_line.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
                ###
                if model.config.batch_size == "chapter":
                    dev_ground_truth = [dp[1] for chapter in model.dev_data for dp in chapter]
                else:
                    dev_ground_truth = [dp[1] for dp in model.dev_data]
                confusion = calculate_confusion(config, predictions, dev_ground_truth)
                print_confusion(confusion, model.speakers.index_to_speaker, model.index_to_weight)
                print 'Total time: {}'.format(time.time() - start)

            saver.restore(session, './weights/whose_line.weights')
            print 'Test'
            print '=-=-='
            test_loss, predictions = model.predict(session, model.test_data)
            print 'Best validation loss: {}'.format(best_val_loss)
            print 'Test loss: {}'.format(test_loss)
            if model.config.batch_size == "chapter":
                test_ground_truth = [dp[1] for chapter in model.test_data for dp in chapter]
            else:
                test_ground_truth = [dp[1] for dp in model.test_data]
            confusion = calculate_confusion(config, predictions, test_ground_truth)
            print_confusion(confusion, model.speakers.index_to_speaker, model.index_to_weight)

            _, predictions = model.predict(session, model.test_data, disallow_other=True)
            confusion = calculate_confusion(config, predictions, test_ground_truth, disallow_other=True)
            print_confusion(confusion, model.speakers.index_to_speaker, model.index_to_weight, disallow_other=True)


if __name__ == "__main__":
    test()