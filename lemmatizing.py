from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

print(lemma.lemmatize("cats"))
print(lemma.lemmatize("cacti"))
print(lemma.lemmatize("geese"))
print(lemma.lemmatize("rocks"))
print(lemma.lemmatize("python"))
print(lemma.lemmatize("better", pos="a"))
print(lemma.lemmatize("best", pos="a"))
print(lemma.lemmatize("run"))
print(lemma.lemmatize("run",'v'))
print(lemma.lemmatize("runing"))
      
