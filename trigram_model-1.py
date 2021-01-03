import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Summer 2012 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    out=list()
    if n==1:
        sequence.insert(0,'START')
    else:
        for i in range(n-1):
            sequence.insert(0,'START')
    sequence.append('STOP')
    for i in range(len(sequence)-n+1):
        out.append(tuple(sequence[i:i+n]))
    return out


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 
        
        for sentence in corpus:
            unigram=get_ngrams(sentence, 1)
            for uni in unigram:
                self.unigramcounts[uni]+=1
                
            bigram=get_ngrams(sentence, 2)
            for bi in bigram:
                self.bigramcounts[bi]+=1
                    
            trigram=get_ngrams(sentence, 3)
            for tri in trigram:
                self.trigramcounts[tri]+=1
                if tri[:2]==('START','START'):
                    self.bigramcounts[('START','START')]+=1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram[:2]]==0:
            return 0.0
        else:
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[:1]]==0:
            return 0.0
        else:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        self.total=sum(self.unigramcounts.values())-self.unigramcounts[('START',)]-self.unigramcounts[('STOP',)]
        if self.total==0:
            return 0.0
        else:
            return self.unigramcounts[unigram]/self.total

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        temp=['START','START']
        tris=list()
        probs=list()
        words=list()
        result=list()
        for i in range(t):
            if temp[1]=='STOP':
                return result
            else:
                for tri in self.trigramcounts.keys():
                    if tri[0]==temp[0] and tri[1]==temp[1]:
                        tris.append(tri)
                        probs.append(self.raw_trigram_probability(tri))
                        words.append(tri[2])
                word=np.random.choice(words,weights=probs,k=1)
                result.append(word)
                temp[0]=temp[1]
                temp[1]=word                
                        
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        smooth=lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:])+lambda3*self.raw_unigram_probability(trigram[2:])
        return smooth
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram=get_ngrams(sentence,3)
        prob=0
        for tri in trigram:
            if self.smoothed_trigram_probability(tri)>0:
                prob+=math.log2(self.smoothed_trigram_probability(tri))
        return prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        prob=0
        m=0
        for sentence in corpus:
            prob+=self.sentence_logprob(sentence)
            m+=len(sentence)
            
        return 2**(-prob/m)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1<pp2:
                correct+=1
            total+=1
    
        for f in os.listdir(testdir2):
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp1<pp2:
                correct+=1
            total+=1
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('Downloads/hw1_data/ets_toefl_data/train_high.txt', 'Downloads/hw1_data/ets_toefl_data/train_low.txt', 'Downloads/hw1_data/ets_toefl_data/test_high', 'Downloads/hw1_data/ets_toefl_data/test_low')
    # print(acc)

