import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes =votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

docs = []
for r in short_pos.split("\n"):
    docs.append( (r, "pos") )

for r in short_neg.split("\n"):
    docs.append( (r, "neg") )

all_words = []

short_pos_words =word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in docs]
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]
#Classifiers
classifier = nltk.NaiveBayesClassifier.train(training_set)
##classifier_f = open("Naive_Bayes_Classifier.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()
print("Classifier Accuracy Present:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinimialNB Accuracy Present:",nltk.classify.accuracy(MNB_classifier, testing_set))


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB Accuracy Present:",nltk.classify.accuracy(BNB_classifier, testing_set))

##from sklearn.linear_model import LogisticRegression, SGDClassifier
##from sklearn.svm import LinearSVC, NuSVC

LR_classifier = SklearnClassifier(LogisticRegression(tol = 1e-4, max_iter = 5))
LR_classifier.train(training_set)
print("Logistics Regression Accuracy Present:",nltk.classify.accuracy(LR_classifier, testing_set))


SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGDClassifier Accuracy Present:",nltk.classify.accuracy(SGD_classifier, testing_set))


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Accuracy Present:",nltk.classify.accuracy(LinearSVC_classifier, testing_set))


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Accuracy Present:",nltk.classify.accuracy(NuSVC_classifier, testing_set))


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LR_classifier,
                                  SGD_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier Accuracy Present:",nltk.classify.accuracy(voted_classifier, testing_set))

                              
##print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
##print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
##print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
##print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
##print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
##print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)                                 
##                                  
                                  
