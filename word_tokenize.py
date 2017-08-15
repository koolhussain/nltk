from nltk.tokenize import sent_tokenize, word_tokenize

Example_Text = "This is just one minor example, but imagine every word in the English language, every possible tense and affix you can put on a word. Having individual dictionary entries per version would be highly redundant and inefficient, especially since, once we convert to numbers, the "'value'" is going to be identical."

print(sent_tokenize(Example_Text))

print(word_tokenize(Example_Text))
