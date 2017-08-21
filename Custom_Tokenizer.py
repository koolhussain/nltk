import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("E:\\python\\nltk\\Obama.txt")
sample_text = state_union.raw("E:\\python\\nltk\\Trump.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]: #sample test for only 5 lines
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
