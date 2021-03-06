import nltk
import pickle
import random
from nltk.corpus import movie_reviews

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

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

features = [(find_features(rev), category) for (rev, category) in docs]

training_set = features[:1500]
testing_set = features[1500:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
#saving the Classifier
save_classifier = open("Naive_Bayes_Classifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

print("Classifier Accuracy Present:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
