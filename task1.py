import nltk
import stanza
import helpers

pos_tagger = stanza.Pipeline(processors= "tokenize,pos",tokenize_pretokenized = True)

# first genre
#produces a tuple of tags and words
# [('me', 'PRP'), ('You', 'PRP')]
sent_pos = nltk.corpus.brown.tagged_sents(categories = "mystery", tagset = "universal")
sent_list1 = helpers.listAppend(sent_pos)
# return a list of list containing universal POS tags for each word
# [[PRP], [PRP]]
tagged_pos1 = helpers.get_pos_from_nltk_tagged_sents(sent_pos)
tagged_list1 = pos_tagger(sent_list1)
# return a list of list containing universal POS tags for each word
# [[PRP], [PRP]]
print("Accuracy of genre 'mystery':")
tagged_list1 = helpers.get_pos_from_stanza_output(tagged_list1)
helpers.accuracy(tagged_pos1, tagged_list1)
print( "\n")

# second genre
sent_pos2 = nltk.corpus.brown.tagged_sents(categories = "religion", tagset = "universal")
sent_list2 = helpers.listAppend(sent_pos2)
tagged_pos2 = helpers.get_pos_from_nltk_tagged_sents(sent_pos2)
tagged_list2 = pos_tagger(sent_list2)
tagged_list2 = helpers.get_pos_from_stanza_output(tagged_list2)
print("Accuracy of genre 'religion':")
helpers.accuracy(tagged_pos2, tagged_list2)
print( "\n")

# third genre
sent_pos3 = nltk.corpus.brown.tagged_sents(categories = "science_fiction", tagset = "universal")
sent_list3 = helpers.listAppend(sent_pos3)
tagged_pos3 = helpers.get_pos_from_nltk_tagged_sents(sent_pos3)
tagged_list3 = pos_tagger(sent_list3)
tagged_list3 = helpers.get_pos_from_stanza_output(tagged_list3)
print("Accuracy of genre 'science_fiction':")
helpers.accuracy(tagged_pos3, tagged_list3)
print( "\n")
