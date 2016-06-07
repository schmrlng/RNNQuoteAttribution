from collections import defaultdict
import numpy as np


def load_chapter_split(datasplitpath):
    chaptermap = {}
    with open(datasplitpath) as f:
        for i, l in enumerate(f):
            for ch in l.strip().split(" "):
                chaptermap[ch] = i
    return chaptermap

def data_iterator(orig_data, batch_size=32, chop_limit=1000, shuffle=False):
    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(len(orig_data))
        data = orig_data[indices]
    else:
        data = orig_data
    ###
    total_processed_examples = 0
    if batch_size == "chapter":
        data_length = sum([len(x) for x in data])
        total_steps = len(data)
    else:
        data_length = len(data)
        total_steps = int(np.ceil(len(data) / float(batch_size)))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        if batch_size == "chapter":
            data_batch = data[step]
        else:
            batch_start = step * batch_size
            data_batch = data[batch_start:batch_start + batch_size]
        X, Y = zip(*data_batch)
        line_length_batch = np.array([len(x) for x in X])
        maxlen = np.max(line_length_batch)
        lines_batch = np.vstack(
            [np.pad(x, (0, maxlen - len(x)), 'constant') for x in X])
        lines_batch = lines_batch[:, :chop_limit]
        labels_batch = Y
        ###
        yield lines_batch, line_length_batch, labels_batch
        total_processed_examples += len(labels_batch)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == data_length, 'Expected {} and processed {}'.format(data_length, total_processed_examples)

class Vocab(object):

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = 'unk'   # TODO: I don't think this is right cf. https://groups.google.com/forum/#!searchin/globalvectors/unknown/globalvectors/9w8ZADXJclA/X6f0FgxUnMgJ
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

class Speakers(object):

    def __init__(self):
        self.speaker_to_index = {}
        self.index_to_speaker = {}
        self.speaker_freq = defaultdict(int)
        self.unknown = 'OTHER'
        self.add_speaker(self.unknown, count=0)

    def add_speaker(self, speaker, count=1):
        if speaker not in self.speaker_to_index:
            index = len(self.speaker_to_index)
            self.speaker_to_index[speaker] = index
            self.index_to_speaker[index] = speaker
        self.speaker_freq[speaker] += count

    def prune(self, count = 5):
        if count > 0:
            sorted_speakers = sorted(self.speaker_freq.iteritems(), key=lambda x: x[1])
            top_speakers = sorted_speakers[-count:]
            top_speakers.append(('OTHER', sum(map(lambda x: x[1], sorted_speakers[:-count]))))
            self.speaker_freq = dict(top_speakers)
            self.index_to_speaker = dict(enumerate(reversed([s[0] for s in sorted(self.speaker_freq.iteritems(), key=lambda x: x[1])]))) # horrible
            self.speaker_to_index = {v:k for k,v in self.index_to_speaker.iteritems()}

    def encode(self, speaker):
        if speaker not in self.speaker_to_index:
            speaker = self.unknown
        return self.speaker_to_index[speaker]

    def decode(self, index):
        return self.index_to_speaker[index]

    def __len__(self):
        return len(self.speaker_freq)

def print_confusion(confusion, num_to_tag, num_to_weight, disallow_other=False):
    """Helper method that prints confusion matrix."""
    if disallow_other:
        for i, tag in num_to_tag.items():   # certainly not the best way to zero out the "OTHER" row, but it works
            if tag == "OTHER":
                confusion[i,:] = 0
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print ('Speaker: {:10} - P {:7.4f} / R {:7.4f} / F1 {:7.4f} \t (loss weight {:6.3f})'.format(tag, prec, recall, 2*prec*recall/(prec+recall), num_to_weight[i])).expandtabs(10)

def calculate_confusion(config, predicted_indices, y_indices, disallow_other=False):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.speaker_count, config.speaker_count), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion