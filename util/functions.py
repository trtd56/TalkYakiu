# coding: utf-8

def generate_batch(generator, batch_size):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sorted_parallel(generator1, generator2, pooling, order=1):
    gen1 = generate_batch(generator1, pooling)
    gen2 = generate_batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x

def fill_batch(batch, h="<s>", t="</s>", n="<non>"):
    max_len = max(len(x) for x in batch)
    return  [[h] + x + [t] + [n] * (max_len - len(x)) for x in batch]

def check_words(text_list, model, is_api=False):
    text_num = model.vocab.word2id(text_list)
    unk_word = [w for i,w in zip(text_num, text_list) if i == 0]
    if len(unk_word) == 0:
        prt = ""
    else:
        prt = ",".join(unk_word)
        if not is_api:
            prt = "\nunknow word: " + prt
    return prt
