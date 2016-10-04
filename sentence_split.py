# coding: utf-8

import sys, glob, math, pickle
import MeCab
import pandas as pd

NOISE_WORDS = ["","『","』","(",")","（","）"]

def read_corpus():
    files = glob.glob('./data/corpus/*')
    if len(files) == 0:
        sys.stderr.write('There is no corpus file.')
        exit()
    else:
        corpus_list = [pd.read_csv(i) for i in files]
        return pd.concat(corpus_list)

def split_sentence(text):
    tagger = MeCab.Tagger("-Owakati")
    text_sp = tagger.parse(text)
    sp_list = text_sp.replace("\n","").split(" ")
    sp_list = del_noise_word(sp_list)
    return sp_list

def del_noise_word(sp_list):
    for noise in NOISE_WORDS:
        if noise in sp_list:
            sp_list.remove(noise)
    return sp_list

if __name__=="__main__":
    corpus = read_corpus()
    genju_req = corpus.loc[:,"0_原住民"]
    yakiu_res = corpus.loc[:,"1_やきう民"]
    req_res_pair = [(split_sentence(g), split_sentence(y)) \
                     for g, y in zip(genju_req, yakiu_res) \
                     if str(type(g)) == "<type 'str'>" and str(type(y)) == "<type 'str'>"]
    with open('./data/split_list/yakiu.pkl', mode='wb') as f:
        pickle.dump(req_res_pair, f)
