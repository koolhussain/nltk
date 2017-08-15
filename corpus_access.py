from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

text = gutenberg.raw("bible-kjv.txt")

token = sent_tokenize(text)

for x in range(5):
    print(token[x])
