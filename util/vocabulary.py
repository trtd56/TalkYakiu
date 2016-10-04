# coding: utf-8

from collections import defaultdict

class Vocabulary:

    def __init__(self):
        pass

    def __len__(self):
        return self.__size

    def stoi(self, s):
        return self.__stoi[s]

    def itos(self, i):
        return self.__itos[int(i)]

    def word2id(self, text):
        return [self.stoi(i) for i in text]

    def id2word(self, text):
        return [self.itos(i) for i in text]

    @staticmethod
    def new(words, min_count=1):
        self = Vocabulary()

        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        is_use = [i >= min_count for i in word_freq.values()]
        tmp = [dic for use, dic in zip(is_use, word_freq.items()) if use]
        word_freq = {k:v for k,v in tmp}

        self.__size = len(word_freq) + 4

        self.__stoi = defaultdict(int)
        self.__stoi['<unk>'] = 0
        self.__stoi['<s>'] = 1
        self.__stoi['</s>'] = 2
        self.__stoi['<non>'] = 3
        self.__itos = [''] * self.__size
        self.__itos[0] = '<unk>'
        self.__itos[1] = '<s>'
        self.__itos[2] = '</s>'
        self.__itos[3] = '<non>'

        for i, (k, v) in zip(range(self.__size - 4), sorted(word_freq.items(), key=lambda x: -x[1])):
            self.__stoi[k] = i + 4
            self.__itos[i + 4] = k

        return self
