#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### REF

# 1. https://stackoverflow.com/questions/34735016/choosing-an-sklearn-pipeline-for-classifying-user-text-data 
# 2. ChatGPT
# 3. https://scikit-learn.org/stable/modules/naive_bayes.html 

from collections import defaultdict
from nltk.corpus import wordnet # This might require "nltk.download('wordnet')" and "nltk.download('omw-1.4')".
import math

import random
from sklearn.model_selection import KFold

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import pathlib

from utils import *

class WSDClassifier(object):
    """
    Abstract class for WSD classifiers
    """

    def evaluate(self, instances):
        """
        Evaluates the classifier on a set of instances.
        Returns the accuracy of the classifier, i.e. the percentage of correct predictions.
        
        instances: list[WSDInstance]
        """
        counter = 0 
        for instance in instances:
            if instance.sense == self.predict_sense(instance):
                counter += 1
        return counter / len(instances)
    
    
        
        pass # TODO

class RandomSense(WSDClassifier):
    """
    RandomSense baseline
    """
    
    def __init__(self):
        pass # Nothing to do.

    def train(self, instances=[]):
        """
        instances: list[WSDInstance]
        """
        
        pass # Nothing to do.

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        
        senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]
        random.shuffle(senses)
        return senses[0]
    
    def __str__(self):
        return "RandomSense"

class MostFrequentSense(WSDClassifier):
    """
    Most Frequent Sense baseline
    """
    
    def __init__(self):
        self.mfs = {} # Should be defined as a dictionary from lemmas to most frequent senses (dict[string -> string]) at training.

    
    def train(self, instances):
        """
        instances: list[WSDInstance]
        """

        senDISTR = sense_distribution(instances) # dict[string -> int]
        
        # print("first get :",senDISTR) # initialized 
        for instance in instances:
            # print(f"instance: {instance},instance.lemma:{instance.lemma}")
            if instance.lemma not in self.mfs:
                self.mfs[instance.lemma] = instance.sense
                
            else:
                if senDISTR[instance.sense] > senDISTR[self.mfs[instance.lemma]]:
                    self.mfs[instance.lemma] = instance.sense

        # print("most frequent:",self.mfs[instance.lemma]) # get the most frequent sense for the lemma
        print("mfs:",self.mfs)
        pass # TODO

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        return self.mfs[instance.lemma]
        
        pass # TODO
    
    def __str__(self):
        return "MostFrequentSense"

class SimplifiedLesk(WSDClassifier):
    """
    Simplified Lesk algorithm
    """
    
    def __init__(self, window_size=-1, idf=False):
        self.signature = {}  # Dictionary from word senses to signatures (dict[word -> dict[sense -> set[string]]])
        self.idf = idf
        self.window_size = window_size
        self.wn_synet = wordnet.synset

    def train(self, instances=[]):
        """
        Train the classifier by building signatures for each sense of each word.
        Each signature is a set containing the definition and examples from WordNet,
        plus the word itself.
        """
        for lemma in WN_CORRESPONDANCES:
         
            for sense in WN_CORRESPONDANCES[lemma]:

                self.signature[sense] = set() # initialize the signature of the sense 

                for synset in WN_CORRESPONDANCES[lemma][sense]:
                    self.signature[sense].update(self.wn_synet(synset).definition().split())

                for example in self.wn_synet(synset).examples():
                    self.signature[sense].update(example.split())

                    self.signature[sense].add(lemma) # because update is to use for multiple 

        ### context 
        for instance in instances:
            if self.window_size == -1:
                self.signature[instance.sense].update(instance.context)
            else:
                self.signature[instance.sense].update(instance.left_context[-self.window_size:]+instance.right_context[:self.window_size])

     
  
    def predict_sense(self, instance):
        ## 2 different ways to get the prediction, one using window size and the other without window size
        # all_senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]
        senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]

        scores = list(len(senses)*[0]) # initialize the scores for each sense
        for i,sense in enumerate(senses):
            for word in instance.context:
                if word in self.signature[sense] and word not in STOP_WORDS:
                    if self.idf:
                        # scores[i] += math.log(len(self.signature)/sum([1 for sense in self.signature if word in self.signature[sense]]))
                        doc_freq = sum(1 for s in self.signature if word in self.signature[s])
                        idf_value = math.log(len(self.signature) / doc_freq) if doc_freq > 0 else 0
                        scores[i] += idf_value
                    else:
                        scores[i] += 1
        mfs = senses[scores.index(max(scores))] # get the sense with the highest score 
        return mfs
        


    
    def __str__(self):
        return "SimplifiedLesk"


def cross_validate(classifier, instances, k=5):
    """`````
    in this cross validation, kf is the KFold object, which is used to split the data into k folds.
    when we set k = 5, we split the data into 5 folds, and we train the model on 4 folds and test on the remaining fold.
    we repeat this process 5 times, each time we use a different fold as the test fold.

    from the index length, we can see that the data is split into 5 folds, and each fold has a different number of instances.

    the score will be then divided by the number of folds to get the average accuracy over the folds.

    for the commented part, it is used to save the train and test instances for debug, so that we can check the instances used in each fold. --not necessary for the final version

    ``` """


    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = []
    for train_index, test_index in kf.split(instances): # split the data k times
        train_instances = [instances[i] for i in train_index] # get the train instances
        test_instances = [instances[i] for i in test_index] # get the test instances
        classifier.train(train_instances) # train on the train instances
        accuracy = classifier.evaluate(test_instances)
        scores.append(accuracy)
        
        
    return scores


class NaiveBayesWSD(WSDClassifier):
    from sklearn.feature_extraction.text import CountVectorizer
    def __init__(self):
        self.stop_words = list(STOP_WORDS) # we use the stopwords that given by the pr,to suit the CountVectorizer, we need to convert the stopwords to a list from a set dict --- SUN
        self.model = make_pipeline(CountVectorizer(stop_words=self.stop_words), MultinomialNB())

    def train(self, instances):
    
        texts = [' '.join(i.context) for i in instances]  # Convert list of words to a single string
        labels = [i.sense for i in instances]
        self.model.fit(texts, labels)

    def predict_sense(self, instance):
        context_str = ' '.join(instance.context)  # Ensure the context is a single string
        return self.model.predict([context_str])[0]

###############################################################



###############################################################

# The body of this conditional is executed only when this file is directly called as a script (rather than imported from another script).
if __name__ == '__main__':
    from twa import WSDCollection
    from optparse import OptionParser
    import pandas as pd 
    import matplotlib.pyplot as plt
    from utils import *


    usage = "Comparison of various WSD algorithms.\n%prog TWA_FILE"
    parser = OptionParser(usage=usage)
    (opts, args) = parser.parse_args()
    if(len(args) > 0):
        sensed_tagged_data_file = args[0]
    else:
        exit(usage + '\nYou need to specify in the command the path to a file of the TWA dataset.\n')

    # Loads the corpus.
    instances = WSDCollection(sensed_tagged_data_file).instances
    
    # Displays the sense distributions.
    pass # TODO
## Task 1: 

    senses =[instance.sense for instance in instances]
    senses = pd.Series(senses)
    senses.value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.savefig('sense_distribution.png')
    plt.show()
    plt.close()
    prettyprint_sense_distribution(instances=instances)

    
    # Evaluation of the random baseline on the whole corpus.
## Task2: 
    pass # TODO
    random_baseline = RandomSense()
    print(f"Random baseline accuracy: {random_baseline.evaluate(instances)}")
            
    
    # Evaluation of the most frequent sense baseline using different splits of the corpus (with `utils.data_split` or `utils.random_data_split`).
## Task3: 
    pass # TODO
    # with random split
    mfs_baseline_rd = MostFrequentSense()
    train_rd,test_rd= random_data_split(instances, p=8, n=10)
    mfs_baseline_rd.train(train_rd)
    # with fixed split
    mfs_baseline_fs = MostFrequentSense()
    train_fs,test_fs= data_split(instances, p=8, n=10)
    mfs_baseline_fs.train(train_fs)
    print(f"Most Frequent Sense baseline accuracy with random split technique: {mfs_baseline_rd.evaluate(test_rd)}, with fixed split technique: {mfs_baseline_fs.evaluate(test_fs)}")

    
   
    
    # Evaluation of Simplified Lesk (with no fixed window and no IDF values) using different splits of the corpus.
## Task4: 
    pass # TODO
    # with fixed split
    clf = SimplifiedLesk()
    train_clf,test_clf = data_split(instances, p=7, n=10)
    clf.train(train_clf)
    # with random split
    clf_rd = SimplifiedLesk()
    train_clf_rd,test_clf_rd = random_data_split(instances, p=7, n=10)
    clf_rd.train(train_clf_rd)
    print(f"Simplified Lesk Accuracy without specifying Window size and IDF=FALSE:{clf.evaluate(test_clf)}, with random split technique: {clf_rd.evaluate(test_clf_rd)}")



    
    # Evaluation of Simplified Lesk (with a window of size 10 and no IDF values) using different splits of the corpus.
##  
    pass # TODO
    # with fixed split
    clf_window= SimplifiedLesk(window_size=10,idf=False)

    clf_window.train(train_clf)
    # with random split
    clf_window_rd = SimplifiedLesk(window_size=10,idf=False)
    clf_window_rd.train(train_clf_rd)
    print(f"Simplified Lesk accuracy with window size fixed at 10 and IDF= FALSE: {clf_window.evaluate(test_clf)}, with random split technique: {clf_window_rd.evaluate(test_clf_rd)}")
    
    # Evaluation of Simplified Lesk (with IDF values and no fixed window) using different splits of the corpus.
## 
    pass # TODO
    # with fixed split
    clf_idf = SimplifiedLesk(idf=True)
    clf_idf.train(train_clf)
    # with random split
    clf_idf_rd = SimplifiedLesk(idf=True)
    clf_idf_rd.train(train_clf_rd)

    print(f"Simplified Lesk accuracy with IDF=TRUE and no fix window size: {clf_idf.evaluate(test_clf)}, with random split technique: {clf_idf_rd.evaluate(test_clf_rd)}")
    
    # Cross-validation
## Task7:  
    pass # 
    lesk_classifier = SimplifiedLesk(window_size=10)
    lesk_scores = cross_validate(lesk_classifier, instances)
    print(f'Average accuracy over {len(lesk_scores)} folds: {sum(lesk_scores) / len(lesk_scores)}')  # calculate the average accuracy over the folds 
    # Naive Bayes classifier 
    pass # TODO
## Instead of from scratch, we adopte SKLEARN to implement the Naive Bayes classifier, please understand the naviebayes general idea --- SUN
    nb_classifier = NaiveBayesWSD()
    nb_classifier.train(train_clf)
    print(f"Naive Bayes classifier accuracy: {nb_classifier.evaluate(test_clf)}")

