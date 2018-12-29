import gzip
import argparse
import logging
import json
import pickle as pkl
import spacy
import itertools

from tqdm import tqdm
from random import randint,shuffle
from collections import Counter


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def data_generator(data):
    with gzip.open(args.input,"r") as f:
        for x in tqdm(f,desc="Reviews",total=count_lines(f)):
            yield json.loads(x)


def to_array_comp(doc):
        return [[w.orth_ for w in s] for s in doc.sents]


def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser,to_array_comp)


def build_dataset(args):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    gen_a,gen_b = itertools.tee(data_generator(args.input),2)
    data = [(z["reviewerID"],z["asin"],tok,z["overall"]) for z,tok in zip(tqdm((z for z in gen_a),desc="reading file"),nlp.pipe((x["reviewText"] for x in gen_b), batch_size=1000000, n_threads=8))]

    print(data[0])
    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    #for sent in data[0][2].sents:
    #    for word in sent:
    #        print(word)

    return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating")}

def build_dataset_debug(args):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    #nlp = spacy.load('en')
    #gen_a,gen_b = itertools.tee(data_generator(args.input),2)
    #data = [(z["reviewerID"],z["asin"],tok,z["overall"]) for z,tok in zip(tqdm((z for z in gen_a),desc="reading file"),nlp.pipe((x["reviewText"] for x in gen_b), batch_size=1000000, n_threads=8))]

    #for token in nlp.pipe((x["reviewText"] for x in data_generator(args.input)), batch_size=1000000, n_threads=8):
    #    print(token)
    gen_a, gen_b = itertools.tee(data_generator(args.input), 2)
    #data = [x for x in nlp.pipe((x["reviewText"] for x in gen_b), batch_size=1000000, n_threads=8)]
    #data2 = [(z["reviewerID"], z["asin"], z["overall"]) for z in tqdm((z for z in gen_a), desc="reading file")]
    data3 = [(z["reviewerID"],z["asin"],tok,z["overall"]) for z,tok in zip(tqdm((z for z in gen_a),desc="reading file"),nlp.pipe((x["reviewText"] for x in gen_b), batch_size=1000000, n_threads=8))]
    print("len(data3): ", len(data3))
    print("len(data3[0]): ", len(data3[0]))
    print("data3[0][2]: ", data3[0][2])
    print("type(data3[0][2]): ", type(data3[0][2]))
    for sent in data3[0][2].sents:
        for word in sent:
            print(word)
    #print(len(data))
    #print(type(data[0]))

    #return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating")}

def main(args):
    ds = build_dataset(args)
    pkl.dump(ds,open(args.output,"wb"))
    #build_dataset_debug(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--nb_splits",type=int, default=5)
    args = parser.parse_args()

    main(args)