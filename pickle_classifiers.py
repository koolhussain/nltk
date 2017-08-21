import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
#from nltk.corpus import movie_reviews
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
all_words = []

#allowing adjectives only
allowed_word_types = ["J"]

for r in short_pos.split("\n"):
    docs.append( (r, "pos") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
        

for r in short_neg.split("\n"):
    docs.append( (r, "neg") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

#save docs
save_docs = open("savedata/docs.pickle","wb")
pickle.dump(docs, save_docs)
save_docs.close()

##short_pos_words =word_tokenize(short_pos)
##short_neg_words = word_tokenize(short_neg)
##
##for w in short_pos_words:
##    all_words.append(w.lower())
##
##for w in short_neg_words:
##    all_words.append(w.lower())
##
###save all_words
##save_all_words = open("savedata/save_all_words.pickle","wb")
##pickle.dump(all_words, save_all_words)
##save_all_words.close()
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

#save word features
save_word_features = open("savedata/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in docs]

save_featuresets = open("savedata/featuresets5k.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#Classifiers
classifier = nltk.NaiveBayesClassifier.train(training_set)
##classifier_f = open("Naive_Bayes_Classifier.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()
print("Classifier Accuracy Present:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("savedata/naivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinimialNB Accuracy Present:",nltk.classify.accuracy(MNB_classifier, testing_set))

save_classifier = open("savedata/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB Accuracy Present:",nltk.classify.accuracy(BNB_classifier, testing_set))

save_classifier = open("savedata/BNB_classifier5k.pickle","wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()


##from sklearn.linear_model import LogisticRegression, SGDClassifier
##from sklearn.svm import LinearSVC, NuSVC

LR_classifier = SklearnClassifier(LogisticRegression(tol = 1e-4, max_iter = 5))
LR_classifier.train(training_set)
print("Logistics Regression Accuracy Present:",nltk.classify.accuracy(LR_classifier, testing_set))

save_classifier = open("savedata/LR_classifier.pickle","wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()


SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGDClassifier Accuracy Present:",nltk.classify.accuracy(SGD_classifier, testing_set))

save_classifier = open("savedata/SGD_classifier5k.pickle","wb")
pickle.dump(SGD_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Accuracy Present:",nltk.classify.accuracy(LinearSVC_classifier, testing_set))

save_classifier = open("savedata/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Accuracy Present:",nltk.classify.accuracy(NuSVC_classifier, testing_set))

save_classifier = open("savedata/NuSVC_classifier5k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LR_classifier,
                                  SGD_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier Accuracy Present:",nltk.classify.accuracy(voted_classifier, testing_set))

                              
                          
                                  
