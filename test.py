from RNNmodels import test

if __name__ == "__main__":
    test(sentencernn = True, contextrnn = True, sentencebi = False, contextbi = False)
    # test(sentencernn = True, contextrnn = False, sentencebi = True, contextbi = True)
    # test(sentencernn = False, contextrnn = True, sentencebi = True, contextbi = True)
    # test(sentencernn = True, contextrnn = True, sentencebi = True, contextbi = False)