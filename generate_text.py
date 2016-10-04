# coding: utf-8

import pickle
import MeCab

from encdec import *
from util.functions import check_words

def main(is_gpu=-1):

    with open('./data/model/sample_model.pkl', mode='rb') as f:
        model = pickle.load(f)
    tagger = MeCab.Tagger("-Owakati")

    while True:
        input_text = raw_input('(´・ω・`) {')
        if input_text == "exit":
            break
        text_sp = tagger.parse(input_text)
        sp_list = text_sp.replace("\n","").split(" ")
        sp_list.remove("")
        text = ["<s>"] + sp_list  + ["</s>"]
        unk_word = check_words(text, model)
        generate_txt = model.predict(text)
        generate_txt.remove("<s>")
        generate_txt.remove("</s>")
        print "彡(ﾟ)(ﾟ)  {" + "".join(generate_txt) + unk_word
        print

if __name__=="__main__":
    main()
