from nltk.corpus import wordnet

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('bush.n.01')
w2 = wordnet.synset('brush.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('cat.n.01')
w2 = wordnet.synset('dog.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('angel.n.01')
w2 = wordnet.synset('devil.n.01')
print(w1.wup_similarity(w2))
