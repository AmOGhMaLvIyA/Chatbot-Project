import nltk
import numpy as np
# nltk.download('punkt')#package that contains pre tokenizing function
from nltk.stem.porter import PorterStemmer#need to see its functioning (done)

stemmer=PorterStemmer()#creating stemming function 

#tokenizing function 
def token(sentence):
    return nltk.word_tokenize(sentence)

#stemming function 
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)#initializing all the values 0 in the bag as to compare it to the position of the word 
    #in case to produce the np array data for training 
    for idx , w  in enumerate(all_words):#idx here is index
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


# testing the bag function 
# sentence=["hello","how","are","you"]
# words=["hi","hello","I","you","bye","thank","cool"]
# bag=bag_of_words(sentence,words)
# print(bag)

# for testing the functioning of tokenizer 
# a=input()
# print(a)
# a=token(a)
# print(a)


# below is the testing of the above created stem function 
# words=["organizes","organize","organizing"]
# stemmed_words=[stem(w) for w in words]
# print(stemmed_words)