import nltk
import stanza
import helpers

pos_tagger = stanza.Pipeline(processors="tokenize,pos",tokenize_pretokenized=True)

# Checking accuracy of sentence 1
sentence1 = "John saw the saw and decided to take it to the table"
sentence_list1 = ["NOUN", "VERB", "DT", "NOUN", "CONJ", "VERB", "TO", "VERB", "PRP", "PREP", "DT", "NOUN"]
Manual_output1 = []
Manual_output1.append(sentence_list1)
stanza_output1 = nltk.pos_tagger(sentence1)
stanza_output1 = helpers.get_pos_from_stanza_output(stanza_output1)
print("Accuracy of pretokenized sentence 1:")
helpers.accuracy(Manual_output1, stanza_output1)
print("\n")

# Checking accuracy of sentence 2
sentence2 = "The dogs bark at the bark"
sentence_list2 = ["DET", "NOUN", "VERB", "DET", "DET", "NOUN"]
Manual_output2 = []
Manual_output2.append(sentence_list2)
stanza_output2 = nltk.pos_tagger(sentence2)
stanza_output2 = helpers.get_pos_from_stanza_output(stanza_output2)
print("Accuracy of pretokenized sentence 2:")
helpers.accuracy(Manual_output2, stanza_output2)
print("\n")

# Checking accuracy of sentence 3
sentence3 = "The baseball pitcher asked for a pitcher of water"
sentence_list3 = ["DET", "NOUN", "NOUN", "VERB", "X", "DET", "NOUN", "X", "NOUN"]
Manual_output3 = []
Manual_output3.append(sentence_list3)
stanza_output3 = nltk.pos_tagger(sentence3)
stanza_output3 = helpers.get_pos_from_stanza_output(stanza_output3)
print("Accuracy of pretokenized sentence 3:")
helpers.accuracy(Manual_output3, stanza_output3)
print("\n")

# Checking accuracy of sentence 4
sentence4 = "The committee chair sat in the center chair"
sentence_list4 = ["DET", "NOUN", "NOUN", "VERB", "X", "DET", "NOUN", "NOUN"]
Manual_output4 = []
Manual_output4.append(sentence_list4)
stanza_output4 = nltk.pos_tagger(sentence4)
stanza_output4 = helpers.get_pos_from_stanza_output(stanza_output4)
print("Accuracy of pretokenized sentence 4:")
helpers.accuracy(Manual_output4, stanza_output4)
print("\n")

# Checking accuracy of sentence 5
sentence5 = "The crane flew over the construction crane"
sentence_list5 = ["DET", "NOUN", "VERB", "X", "DET", "NOUN", "NOUN"]
Manual_output5 = []
Manual_output5.append(sentence_list5)
stanza_output5 = nltk.pos_tagger(sentence5)
stanza_output5 = helpers.get_pos_from_stanza_output(stanza_output5)
print("Accuracy of pretokenized sentence 5:")
helpers.accuracy(Manual_output5, stanza_output5)
print("\n")
