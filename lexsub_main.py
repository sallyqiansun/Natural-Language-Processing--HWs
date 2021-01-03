#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List

from nltk.stem import WordNetLemmatizer
import string
def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def get_candidates(lemma, pos) -> List[str]:
    l=wn.lemmas(lemma,pos=pos)
    out=[]
    sl=[]
    for li in l:
        sl.extend(li.synset().lemmas())
    for sli in sl:
        o=sli.name()
        if o!=lemma:
            if o.find('_')!=-1:
                out.append(o.replace('_',' '))
            else:
                out.append(o)
    return out

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma=context.lemma
    pos=context.pos
    l=wn.lemmas(lemma,pos=pos)
    out=[]
    f={}

    for li in l:
        sl=li.synset().lemmas()
        for sli in sl:
            o=sli.name().lower()
            if o!=lemma:
                if o.find('_')!=-1:
                    o_new=o.replace('_',' ')
                    out.append(o_new)
                    if o_new in f:
                        f[o_new]+=sli.count()
                    else:
                        f[o_new]=sli.count()
                else:
                    out.append(o)
                    if o in f:
                        f[o]+=sli.count()
                    else:
                        f[o]=sli.count()
    return max(f,key=f.get)

def wn_simple_lesk_predictor(context : Context) -> str:
    lemma=context.lemma
    l=wn.lemmas(lemma,pos=context.pos)
    synsets=wn.synsets(lemma)
    definition={}
    example={}
    os=[]

    lcontex=" ".join(context.left_context)
    rcontex=" ".join(context.right_context)
    allcont=set(tokenize(lcontex+rcontex))
    allc=set()
    stop=list(dict.fromkeys(stopwords.words('english')))
    for c in allcont:
        if c not in stop:
            allc.add(c)

    for li in l:
        si=li.synset()
        definition[si]=[si.definition()]
        example[si]=si.examples()

        hyper=si.hypernyms()
        for h in hyper:
            definition[si].append(h.definition())
            example[si].extend(h.examples())

    for key in definition:
        d=definition[key]
        e=example[key]
        words=[]
        for di in d:
            words.extend(tokenize(di))
        for ei in e:
            words.extend(tokenize(ei))
        lem=set()
        for i in range(len(words)):
            if words[i] not in stop:
                lem.add(WordNetLemmatizer().lemmatize(words[i]))
        overlap=allc & lem
        if len(overlap)>0:
            os.append(key)

    if len(os)<0:
        return wn_frequency_predictor(context)
    else:
        sl=[]
        f={}
        for si in os:
            for li in l:
                sl.extend(si.lemmas())
        for sli in sl:
            o=sli.name()
            if o!=lemma:
                if o.find('_')!=-1:
                    o_new=o.replace('_',' ')
                    if o_new in f:
                        f[o_new]+=1
                    else:
                        f[o_new]=1
                else:
                    if o in f:
                        f[o]+=1
                    else:
                        f[o]=1

        if len(f)>0:
            return max(f,key=f.get)
        else:
            return wn_frequency_predictor(context)

     #replace for part 3


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:
        lemma=context.lemma
        pos=context.pos
        stop=stopwords.words('english')
        candidates=get_candidates(lemma, pos=pos)
        sim={}
        for s in candidates:
            if s in self.model.wv.vocab:
                sim[s]=self.model.similarity(lemma, s)
        return max(sim,key=sim.get)
        #replace for part 4


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos
        #i=[]
        sl=get_candidates(lemma=lemma, pos=pos)
        sl_str=' '.join(sl)

        all=[l for l in context.left_context if l not in string.punctuation]
        all.append('[MASK]')
        right=[r for r in context.right_context if r not in string.punctuation]
        all=all+right
        input=' '.join(all)
        input_toks = self.tokenizer.encode(input)
        ls=self.tokenizer.convert_ids_to_tokens(input_toks)
        for i in range(len(ls)):
            if ls[i]=='[MASK]':
                ind=i
                break

        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][ind])[::-1]

        candidate_toks = self.tokenizer.encode(sl_str)

        for i in best_words:
            if i in candidate_toks:
                return self.tokenizer.convert_ids_to_tokens([i])
        #out=self.tokenizer.convert_ids_to_tokens(candidate_toks)
        #return out[0]
         # replace for part 5


# my part 6 is modifying part 3, trying to focus on a smaller range of context,
# i.e. half length of left_context and half length of right_context
def pred(context : Context) -> str:
    synonyms = get_candidates(context.lemma, context.pos)
    lc=context.left_context
    rc=context.right_context
    half_l_len=int(len(lc)/2)
    half_r_len=int(len(rc)/2)
    half_l=lc[half_l_len:]
    half_r=rc[:half_r_len]

    l=wn.lemmas(context.lemma,pos=context.pos)
    synsets=wn.synsets(context.lemma)
    definition={}
    example={}
    os=[]

    allcont=set(half_l+half_r)
    allc=set()
    stop=list(dict.fromkeys(stopwords.words('english')))
    for c in allc:
        if c not in stop:
            allc.add(c)

    for li in l:
        si=li.synset()
        definition[si]=[si.definition()]
        example[si]=si.examples()

        hyper=si.hypernyms()
        for h in hyper:
            definition[si].append(h.definition())
            example[si].extend(h.examples())

    for key in definition:
        d=definition[key]
        e=example[key]
        words=[]
        for di in d:
            words.extend(tokenize(di))
        for ei in e:
            words.extend(tokenize(ei))
        lem=set()
        for i in range(len(words)):
            if words[i] not in stop:
                lem.add(WordNetLemmatizer().lemmatize(words[i]))
        overlap=allc & lem
        if len(overlap)>0:
            os.append(key)

    if len(os)<0:
        return wn_frequency_predictor(context)
    else:
        sl=[]
        f={}
        for si in os:
            for li in l:
                sl.extend(si.lemmas())
        for sli in sl:
            o=sli.name()
            if o!=lemma:
                if o.find('_')!=-1:
                    o_new=o.replace('_',' ')
                    if o_new in f:
                        f[o_new]+=1
                    else:
                        f[o_new]=1
                else:
                    if o in f:
                        f[o]+=1
                    else:
                        f[o]=1

        if len(f)>0:
            return max(f,key=f.get)
        else:
            return wn_frequency_predictor(context)


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).



    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    bert=BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict_nearest(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
