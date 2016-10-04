# coding: utf-8

import pickle
import copy
import numpy
import pandas as pd

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, cuda

from util.vocabulary import Vocabulary
from util.functions import fill_batch, generate_batch, sorted_parallel

N_EPOCH = 30
N_BATCH = 100
EMBED_SIZE = 300
HIDDEN_SIZE = 300
GPU = -1

xp = cupy.cupy if GPU >= 0 else numpy

def grep_vocabulary(words):
    w_list = []
    for w in [w[0] + w[1] for w in words]:
        w_list.extend(w)
    return w_list

def divide_rep_res(req_res_list):
    req = [i[0] for i in req_res_list]
    res = [i[1] for i in req_res_list]
    return req, res

def parse_batch(batch, is_rev=False):
    return ["".join(i) for i in batch]

class Encoder(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size, ignore_label):
        super(Encoder, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=ignore_label),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size),
        )
        self.ignore_label = ignore_label

    def __call__(self, x, c_prev, h_prev):
        e = F.tanh(self.xe(x))
        c, h = F.lstm(c_prev, self.eh(e) + self.hh(h_prev))
        tmp = x != self.ignore_label
        enable = Variable(xp.tile(tmp.reshape(len(x), 1), len(c.data[0])))
        c = F.where(enable, c, c_prev)
        h = F.where(enable, h, h_prev)
        return c, h

class Decoder(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size),
            hf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size),
        )

    def __call__(self, y, c, h):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        f = F.tanh(self.hf(h))
        return self.fy(f), c, h

class EncoderDecoder(Chain):

    def __init__(self, w_list, embed_size, hidden_size, is_gpu, ignore_label=3):
        self.vocab = Vocabulary().new(w_list)
        self.vocab_size = len(self.vocab)
        super(EncoderDecoder, self).__init__(
            enc = Encoder(self.vocab_size, embed_size, hidden_size, ignore_label),
            dec = Decoder(self.vocab_size, embed_size, hidden_size),
        )
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_gpu = is_gpu
        if self.is_gpu >= 0:
            self.to_gpu()
        self.optimizer = optimizers.SMORMS3()
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    def reset(self, batch_size):
        self.zerograds()
        self.c = Variable(xp.zeros((batch_size, self.hidden_size), dtype=xp.float32))
        self.h = Variable(xp.zeros((batch_size, self.hidden_size), dtype=xp.float32))

    def encode(self, x):
        self.c, self.h = self.enc(x, self.c, self.h)

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

    def forward(self, req_batch, res_batch):
        req_len = len(req_batch[0])
        res_len = len(res_batch[0])
        batch_size = len(req_batch)

        self.reset(batch_size)
        x = xp.array([2 for _ in range(batch_size)], dtype=xp.int32)
        self.encode(x)

        for l in reversed(range(req_len)):
            x = xp.array([req_batch[k][l] for k in range(batch_size)], dtype=xp.int32)
            self.encode(x)

        loss = 0
        out_list = []
        t = xp.array([1 for _ in range(batch_size)], dtype=xp.int32)
        for l in range(res_len):
            y = self.decode(t)
            t = xp.array([res_batch[k][l] for k in range(batch_size)], dtype=xp.int32)
            loss += F.softmax_cross_entropy(y, t)
            if self.is_gpu >= 0:
                out = cuda.to_cpu(y.data.argmax(1))
            else:
                out = y.data.argmax(1)
            out_list.append(out)
        hyp_batch = xp.array([out_list[i] for i in range(res_len)], dtype=xp.int32).T
        loss.backward()
        self.optimizer.update()
        loss.unchain_backward()
        return hyp_batch, loss.data


    def predict(self, text, limit=100):
        text = self.vocab.word2id(text)

        self.reset(1)
        x = xp.array([2], dtype=xp.int32)
        self.encode(x)

        for l in reversed(range(len(text))):
            x = xp.array([text[l]], dtype=xp.int32)
            self.encode(x)

        out_list = []
        t = xp.array([1], dtype=xp.int32)
        for l in range(limit):
            y = self.decode(t)
            if self.is_gpu >= 0:
                out = cuda.to_cpu(y.data.argmax(1))
            else:
                out = y.data.argmax(1)
            out_list.append(out)
            if out[0] == 2:
                break
            t = xp.array(out, dtype=xp.int32)

        return self.vocab.id2word(out_list)

def main(n_epoch=N_EPOCH, n_batch=N_BATCH, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, is_gpu=GPU):

    with open('./data/split_list/yakiu.pkl', mode='rb') as f:
        req_res_list = pickle.load(f)

    #req_res_list = req_res_list[:10]
    req, res = divide_rep_res(req_res_list)
    w_list = grep_vocabulary(req_res_list)

    model = EncoderDecoder(w_list, embed_size, hidden_size, is_gpu)

    for epoch in range(n_epoch):
        batch = generate_batch(sorted_parallel(req, res, 1 * n_batch), n_batch)

        loss_list = []
        for req_batch, res_batch in batch:
            req_batch = fill_batch(req_batch)
            res_batch = fill_batch(res_batch)

            req_batch = xp.array([model.vocab.word2id(i) for i in req_batch], dtype=xp.int32)
            res_batch = xp.array([model.vocab.word2id(i) for i in res_batch], dtype=xp.int32)

            hyp_batch, loss = model.forward(req_batch, res_batch)
            loss_list.append(loss)

        print epoch,"\t",sum(loss_list)/float(len(loss_list))

    eval_df = []
    for one_req, one_res in zip(req, res):
        one_req = ["<s>"] + one_req + ["</s>"]
        one_res = ["<s>"] + one_res + ["</s>"]
        generate_txt = model.predict(one_req)
        eval_df.append({"原住民":"".join(one_req),"やきう民":"".join(one_res),"生成文":"".join(generate_txt)})

    pd.DataFrame(eval_df).to_csv("./data/evaluation/eval01.csv") 

    if is_gpu >= 0:
        model.is_gpu = -1
    with open('./data/model/sample_model.pkl', mode='wb') as f:
        pickle.dump(copy.deepcopy(model).to_cpu(), f)

if __name__=="__main__":
    main()
