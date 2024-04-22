#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet # This might require "nltk.download('wordnet')" and "nltk.download('omw-1.4')".
import math
import random

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
    
    def __init__(self,window_size=-1,idf=False):
        """
        """
        
        self.signatures = {} # Should be defined as a dictionary from senses to signatures (dict[string -> set[string]]) at training.
        self.idf = idf
        self.window_size = window_size
    def train(self, instances=[]):
        """
        instances: list[WSDInstance]

        {'bass': {'bass%music': ['bass.n.01', 'bass.n.02', 'bass.n.03', 'bass.n.06', 'bass.n.07'], 'bass%fish': ['sea_bass.n.01', 'freshwater_bass.n.01', 'bass.n.08']}, 'crane': {'crane%machine': ['crane.n.04'], 'crane%bird': ['crane.n.05']}, 'motion': {'motion%physical': ['gesture.n.02', 'movement.n.03', 'motion.n.03', 'motion.n.04', 'motion.n.06'], 'motion%legal': ['motion.n.05']}, 'palm': {'palm%hand': ['palm.n.01'], 'palm%tree': ['palm.n.03']}, 'plant': {'plant%factory': ['plant.n.01'], 'plant%living': ['plant.n.02']}, 'tank': {'tank%vehicle': ['tank.n.01'], 'tank%container': ['tank.n.02']}}
        {'bass%music': ['bass.n.01', 'bass.n.02', 'bass.n.03', 'bass.n.06', 'bass.n.07'], 'bass%fish': ['sea_bass.n.01', 'freshwater_bass.n.01', 'bass.n.08']}
        ['sea_bass.n.01', 'freshwater_bass.n.01', 'bass.n.08']
        ['bass.n.01', 'bass.n.02', 'bass.n.03', 'bass.n.06', 'bass.n.07']
    

        """
    # without idf and windowsize 
        for instance in instances:
            correspondances = WN_CORRESPONDANCES[instance.lemma][instance.sense]
            for corres in correspondances:
                if instance.lemma not in self.signatures:
                    self.signatures[instance.lemma] = {}
                if instance.sense not in self.signatures[instance.lemma]:
                    self.signatures[instance.lemma][instance.sense] = set()
                self.signatures[instance.lemma][instance.sense].add(wordnet.synset(corres).definition())
                #get example
                self.signatures[instance.lemma][instance.sense].update(wordnet.synset(corres).examples()) # update 
                self.signatures[instance.lemma][instance.sense].add(instance.lemma)
    # with window size
        
        
        # For the signature of a sense, use (i) the definition of each of the corresponding WordNet synsets, (ii) all of the corresponding examples in WordNet and (iii) the corresponding training instances.
        pass # TODO

    def predict_sense(self, instance,window_size=-1):
        """
        instance: WSDInstance
        """
        ## without window
        sigatures = self.signatures[instance.lemma][instance.sense]
        if window_size == -1:
            max_overlap = 0
            best_sense = None
            for sense in self.signatures[instance.lemma]:
                overlap = len(sigatures.intersection(self.signatures[instance.lemma][sense]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = sense
                
        # with window
        if window_size>0:
            window_size = self.window_size
            context = instance.context
            context = context.split()
            index = context.index(instance.lemma)
            context = context[max(0, index - window_size):min(len(context), index + window_size + 1)]
            context = ' '.join(context)
            sigatures = self.signatures[instance.lemma][instance.sense]
            max_overlap = 0
            best_sense = None
            for sense in self.signatures[instance.lemma]:
                overlap = len(sigatures.intersection(self.signatures[instance.lemma][sense]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = sense
            
        return best_sense
    
        
        pass # TODO
    
    def __str__(self):
        return "SimplifiedLesk"

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
    simplified_lesk = SimplifiedLesk()
    simplified_lesk.train(train)

    print(f"Simplified Lesk accuracy: {simplified_lesk.evaluate(test)}")


    
    # Evaluation of Simplified Lesk (with a window of size 10 and no IDF values) using different splits of the corpus.
    pass # TODO
    simplified_lesk = SimplifiedLesk(window_size=10)
    simplified_lesk.train(train)
    print(f"Simplified Lesk accuracy with window size 10: {simplified_lesk.evaluate(test)}")
    
    # Evaluation of Simplified Lesk (with IDF values and no fixed window) using different splits of the corpus.
    pass # TODO
    
    # Cross-validation
    pass # TODO
    
    # Naive Bayes classifier
    pass # TODO
