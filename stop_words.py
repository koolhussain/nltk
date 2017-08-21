from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

TEXT="For example, you may wish to completely cease analysis if you detect words that are commonly used sarcastically, and stop immediately. Sarcastic words, or phrases are going to vary by lexicon and corpus. For now, we'll be considering stop words as words that just contain no meaning, and we want to remove them."

tokens = word_tokenize(TEXT)
stop_words = set(stopwords.words('english'))

filter_sent = []

for w in tokens:
    if w not in stop_words:
        filter_sent.append(w)

print(tokens)
print(filter_sent)
