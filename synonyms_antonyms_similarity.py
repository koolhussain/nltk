from nltk.corpus import wordnet

syn = wordnet.synsets("Bad")
#for the 1st word
print(syn[0].name())
#for all the words
for s in syn:
    print(s.name())

for sy in syn:
    lem = sy.lemmas()
    print(lem)#[Lemma('bad.n.01.bad'), Lemma('bad.n.01.badness')]
    for l in lem:
        print(l.name())#bad badness

#defination of synonyms and examples
for s in syn:
    print(s.name())
    print(s.definition())
    print(s.examples())
