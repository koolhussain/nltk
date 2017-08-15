import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("E:\\python\\nltk\\Obama.txt")
sample_text = state_union.raw("E:\\python\\nltk\\Trump.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized =custom_sent_tokenizer.tokenize(sample_text)

def content_process():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

content_process()
