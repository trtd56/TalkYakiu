# coding: utf-8

from flask import Flask, jsonify, g
import MeCab
import pickle
import os

from encdec import *
from util.functions import check_words

app = Flask(__name__)

@app.route('/')
def index():
    return "よろしくニキーww"

@app.route('/talk/<text>')
def hello_world(text):
    yakiu, unk_word = talk(text)
    result = {"Yakiu": yakiu, "unk": unk_word}
    return jsonify(ResultSet=result)

@app.before_request
def before_request():
    with open('./data/api/model', mode='rb') as f:
        g.model = pickle.load(f)
    g.tagger = MeCab.Tagger("-Owakati")

def talk(input_text):
    text_sp = g.tagger.parse(str(input_text.encode('utf-8')))
    sp_list = text_sp.replace("\n","").split(" ")
    sp_list.remove("")
    #unk_word = check_words(sp_list, model, is_api=True)
    unk_word = check_words(sp_list, g.model, is_api=True)
    text = ["<s>"] + sp_list  + ["</s>"]
    generate_txt = g.model.predict(text)
    generate_txt.remove("<s>")
    generate_txt.remove("</s>")
    return "".join(generate_txt), unk_word

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
