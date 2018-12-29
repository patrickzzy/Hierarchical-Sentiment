import gzip
import json
import spacy
import pickle as pkl
from tqdm import tqdm
from random import randint

def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0) 
    return count

def test_prepare_data():
    with gzip.open('reviews_Amazon_Instant_Video_5.json.gz', 'r') as f:
        for x in tqdm(f, desc="Reviews", total=count_lines(f)):
            obj = json.loads(x)
            print(obj)
            break

def dump_small_data():
    nb_splits = 5
    datadict = pkl.load(open('prepared_data.bin', 'rb'))
    datadict['data'] = datadict['data'][:100]
    splits = [randint(0,nb_splits-1) for _ in range(0,len(datadict['data']))]
    datadict['splits'] = splits
    pkl.dump(datadict, open('test_data.bin', 'wb'))

def test_han():
    #datadict = pkl.load(open('prepared_data.bin', 'rb'))
    datadict = pkl.load(open('test_data.bin', 'rb'))
    data = datadict['data']
    print("data[0]: ", data[0])
    print("type(data[0][2]): ", type(data[0][2]))
    print("data[0][2]: ", data[0][2])
    for sent in data[0][2]:
        print(sent)
        #for word in sent:
        #    print(word)

def to_array_comp(doc):
    for sent in doc.sents:
        print("In to_array_comp: %s" % (sent))

def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser, to_array_comp)

def test_spacy():
    review_text = "I had big expectations because I love English TV, in particular Investigative and detective stuff but this guy is really boring. It didn't appeal to me at all."
    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    print("nlp.pipe_names are: ", nlp.pipe_names)
    doc_pipe = nlp.pipe(review_text.decode())
    print("type(doc_pipe): ", type(doc_pipe))
    doc_call = nlp(review_text.decode())
    print("type(doc_call): ", type(doc_call))
    print(doc_call)
    #doc = nlp(review_text.decode())
    #for token in doc:
    #    print(token)


def main():
    #test_prepare_data()
    test_han() 
    #test_spacy()
    #dump_small_data()

if __name__ == '__main__':
    main()
