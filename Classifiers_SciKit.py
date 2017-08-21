import nltk
import pickle
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

docs = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append((list(movie_reviews.words(fileid)), category))

random.shuffle(docs)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

features = [(find_features(rev), category) for (rev, category) in docs]

training_set = features[:1500]
testing_set = features[1500:]
#Classifiers

classifier_f = open("Naive_Bayes_Classifier.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
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


                                  
