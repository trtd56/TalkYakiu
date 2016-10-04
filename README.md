# About

- This program is scraping the thread of 2ch.
- To create a pair of conversations Yakiu-min and Genju-min.

# Directory structure

~~~
/
├corpus_generator.py
├sentence_split.py
└data/
 ├corpus/
 └split_list/
~~~

# How to use

## generate corpus

- make url_list.csv in data directory

~~~
url(header),title(header)
{thread url},{title name}
~~~

- run this script

~~~
$ python corpus_generate.py
~~~

## make split sentence pickle

~~~
$ python sentence_split.py
~~~

## generate encdec model

~~~
$ python encdec.py
~~~

## talk Yakiu-min

~~~
$ python generate_text.py
~~~
