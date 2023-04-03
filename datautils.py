import pdb

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model,cache_dir):
    print("get_wikitext2")
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1',cache_dir='/datasets/tmp/wikitext/', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1',cache_dir='/datasets/tmp/wikitext/', split='test')

    from transformers import AutoTokenizer
    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model,cache_dir):
    print("get_ptb")
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank',cache_dir='/datasets/tmp/ptb_text_only/', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank',cache_dir='/datasets/tmp/ptb_text_only/', split='validation')

    from transformers import AutoTokenizer
    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model,cache_dir):
    print("get_c4")
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', cache_dir='/datasets/tmp/allenai--c4/', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', cache_dir='/datasets/tmp/allenai--c4/',data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',cache_dir=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model,cache_dir)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model,cache_dir)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model,cache_dir)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model,cache_dir)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model,cache_dir)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model,cache_dir)
        train=wiki_train+ptb_train+c4_train
        val=None
        return train,val
