#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from nltk.corpus import wordnet # This might require "nltk.download('wordnet')" and "nltk.download('omw-1.4')".
import math
import random
from sklearn.model_selection import KFold

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

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
        for instance in instances:
            if instance.lemma not in self.mfs:
                self.mfs[instance.lemma] = instance.sense
            else:
                if senDISTR[instance.sense] > senDISTR[self.mfs[instance.lemma]]:
                    self.mfs[instance.lemma] = instance.sense
        # by this we are meaning that the most frequent sense is the one that appears the most in the training data, so we are going to use the sense that appears the most in the training data as the most frequent sense for the lemma
        #这个训练时代表的是，对于每个lemma，我们找到了在训练数据中出现最多的sense，然后将其作为这个lemma的最频繁sense，也就是说，我们认为这个sense是这个lemma的最频繁sense，然后在预测时，我们就直接返回这个sense
        
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
            # print(lemma)  ##uncomment for debug ----sun
            for sense in WN_CORRESPONDANCES[lemma]:
                # print(sense) ##uncomment for debug ----sun
                self.signature[sense] = set() # initialize the signature of the sense 
                # DEFINITION USING UPDATE INSTEAD OF ADD BECAUSE WE WANT TO ADD MULTIPLE ELEMENTS --sun
                for synset in WN_CORRESPONDANCES[lemma][sense]:
                    self.signature[sense].update(self.wn_synet(synset).definition().split())
                # print(signatures[sense]) ##uncomment for debug ----sun
                # EXAMPLE USING UPDATE INSTEAD OF ADD BECAUSE WE WANT TO ADD MULTIPLE ELEMENTS --sun
                for example in self.wn_synet(synset).examples():
                    self.signature[sense].update(example.split())
                # ADD LEMMA by using add instead of update because we want to add only one element --- sun
                    self.signature[sense].add(lemma) # because update is to use for multiple 
                # print(self.signature[sense]) ##uncomment for debug ----sun
        ### context 
        for instance in instances:
            if self.window_size == -1:
                self.signature[instance.sense].update(instance.context)
            else:
                self.signature[instance.sense].update(instance.left_context[-self.window_size:]+instance.right_context[:self.window_size])
            # print(self.signature) ##uncomment for debug ----sun

     
  
    def predict_sense(self, instance):
        ## 2 different ways to get the prediction, one using window size and the other without window size
        
        all_senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]
        
        # mfs = all_senses[0]  # Assume the first sense in the list is the most frequent
        scores = list(len(senses)*[0]) # initialize the scores for each sense
        for i,sense in enumerate(senses):
            for word in instance.context:
                if word in self.signature[sense] and word not in STOP_WORDS:
                    if self.idf:
                        scores[i] += math.log(len(self.signature[sense])/sum([1 for sense in self.signature if word in self.signature[sense]]))
                    else:
                        scores[i] += 1
        mfs = senses[scores.index(max(scores))] # get the sense with the highest score 

        return mfs
        


    
    def __str__(self):
        return "SimplifiedLesk"

def cross_validate(classifier, instances, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(instances):
        train_instances = [instances[i] for i in train_index]
        test_instances = [instances[i] for i in test_index]

        classifier.train(train_instances)
        accuracy = classifier.evaluate(test_instances)
        scores.append(accuracy)

    return scores

class NaiveBayesWSD(WSDClassifier):
    def __init__(self):
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())

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

    senses =[instance.sense for instance in instances]
    senses = pd.Series(senses)
    senses.value_counts().plot(kind='bar')
    # plt.show()
    prettyprint_sense_distribution(instances=instances)

    
    # Evaluation of the random baseline on the whole corpus.

    pass # TODO
    random_baseline = RandomSense()
    # counter = 0
    # for instance in instances:
    #     if instance.sense == random_baseline.predict_sense(instance):
    #         counter += 1
    # accuracy = counter / len(instances)
    # print("Random baseline accuracy by calculating: ", accuracy)
    print(f"Random baseline accuracy: {random_baseline.evaluate(instances)}")
            
    
    # Evaluation of the most frequent sense baseline using different splits of the corpus (with `utils.data_split` or `utils.random_data_split`).
    pass # TODO
    mfs_baseline = MostFrequentSense()
    train,test= data_split(instances, p=8, n=10)
    mfs_baseline.train(train)
    print(f"Most Frequent Sense baseline accuracy: {mfs_baseline.evaluate(test)}")
    
   
    
    # Evaluation of Simplified Lesk (with no fixed window and no IDF values) using different splits of the corpus.
    pass # TODO
    # split 
    # train,test = random_data_split(instances, p=8, n=10)
    # if we keep the same train , it will easily be 1 at acc so we many change something
    # for i in range(10):
    #     train,test = random_data_split(instances, p=i, n=10)
    #     clf = SimplifiedLesk()
    #     clf.train(train)
    #     print(f"Simplified Lesk Accuracy without specifying Window size and IDF=FALSE:{clf.evaluate(test)}"
    clf = SimplifiedLesk()
    train_clf,test_clf = data_split(instances, p=7, n=10)
    clf.train(train_clf)
    print(f"Simplified Lesk Accuracy without specifying Window size and IDF=FALSE:{clf.evaluate(test_clf)}")



    
    # Evaluation of Simplified Lesk (with a window of size 10 and no IDF values) using different splits of the corpus.
    pass # TODO

    clf_window= SimplifiedLesk(window_size=10,idf=False)

    clf_window.train(train_clf)
    print(f"Simplified Lesk accuracy with window size fixed at 10 and IDF= FALSE: {clf_window.evaluate(test_clf)}")
    
    # Evaluation of Simplified Lesk (with IDF values and no fixed window) using different splits of the corpus.
    pass # TODO
    clf_idf = SimplifiedLesk(idf=True)
    clf_idf.train(train_clf)
    print(f"Simplified Lesk accuracy with IDF=TRUE and no fix window size: {clf_idf.evaluate(test_clf)}")
    
    # Cross-validation
    pass # TODO
    lesk_classifier = SimplifiedLesk(window_size=10)
    lesk_scores = cross_validate(lesk_classifier, instances)
    print(f'Average accuracy over {len(lesk_scores)} folds: {sum(lesk_scores) / len(lesk_scores)}') 
    # Naive Bayes classifier
    pass # TODO
    nb_classifier = NaiveBayesWSD()
    nb_classifier.train(train_clf)
    print(f"Naive Bayes classifier accuracy: {nb_classifier.evaluate(test_clf)}")
