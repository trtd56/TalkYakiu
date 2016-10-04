# coding: utf-8

import requests, re
import pandas as pd

def split_talk(talk):
    lines = talk.split("<br>")
    match_lines = [re.search("「(.*?)」$",i) for i in lines]
    check_chara = ["y" if i.find("彡") else "g" for i in lines]
    return [(c, re.sub("「|」","",i.group())) for c, i in zip(check_chara,match_lines) if i is not None]

def make_pair(pair, talk_list):
    if talk_list:
        talk = talk_list.pop(0)
        tmp = [(talk[i-1], talk[i]) for i in range(len(talk)) if i > 0]
        pair.extend([i for i in tmp if i[0][0] == "y" and i[1][0] == "g"])
        return make_pair(pair, talk_list)
    else:
        return pair

def scraping(url):
    html = requests.get(url).text
    comments = [i for i in html.split("\n") if i.find("<dd class=") != -1]
    talk = [i.encode('utf-8') for i in comments if i.find(u"彡") != -1 and i.find(u"ω")]
    return [split_talk(i) for i in talk]

def generate_corpus(url, title):
    talk_list = scraping(url)
    pair_list = make_pair([], talk_list)
    out = {"0_原住民":[],"1_やきう民":[]}
    for pair in pair_list:
        out["0_原住民"].append(pair[0][1])
        out["1_やきう民"].append(pair[1][1])
    pd.DataFrame(out).to_csv('./data/corpus/'+title+'.csv', index=False)

if __name__=="__main__":
    url_data = pd.read_csv("./data/url_list.csv")
    url_data = zip(list(url_data.iloc[:,0]),list(url_data.iloc[:,1]))
    for u,t in url_data:
        generate_corpus(u,t)

