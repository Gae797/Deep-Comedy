# -*- coding: utf-8 -*-

'''
This module creates the vocabulary file used by the RhymeTokenizer.
'''

import pickle
import os.path
import Preprocessor

pkl_file = "Dante/dantes_dictionary.pkl"
vocab_file_name = "./Vocab/RhymeVocab"

alphabet_chars = ["a","b","c","d","e","f","g","h","i","j","l",
                  "m","n","o","p","q","r","s","t","u","v","x","y","z"]
special_chars = ["'","à","ä","è","é","ë","ì","ï","ò","ó","ö","ù","ü"]
special_tokens = ["PAD","SLA","ELA","SLB","ELB","SLC","ELC"]

vowels=["a","e","i","o","u","à","ä","è","é","ë","ì","ï","ò","ó","ö","ù","ü"]

def openPKL(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data
    
def getRhymePosition(entry):
    
    list_ = entry[0]
    syllb_acc = list_[0][2]
    syllb_word = list_[1].replace("_","")
    
    counter=0
    for i in range(len(syllb_word)-1,-1,-1):
        
        if counter==abs(syllb_acc) and syllb_word[i] in vowels and (i==0 or syllb_word[i-1] not in vowels):
            return i, syllb_word
        
        if syllb_word[i]=="|":
            counter+=1
            
def splitWord(entry):
    
    '''
    This function splits each word of the vocabulary into two parts:
    the rhyming component and the rest of the word.
    '''
    
    i, syllb_word = getRhymePosition(entry)
    
    base_part = ""
    rhyme_part = ""
    
    if i==0:
        base_part = syllb_word[0:].replace("|","")
        rhyme_part = ""
        
    else:
        for c in range(len(syllb_word)):
            
            if c<i:
                base_part+=syllb_word[c]
            else:
                rhyme_part+=syllb_word[c]
                
        base_part = base_part.replace("|","")
        rhyme_part = "##" + rhyme_part.replace("|","")
    
    return (base_part,rhyme_part)

def containVowels(word):
    
    if word=="":
        return False
    
    for vowel in vowels:
        if vowel in word:
            return True
        
    return False

def createVocab():
    
    data = openPKL(pkl_file)
    
    vocab = [splitWord(data[entry]) for entry in data if  containVowels(data[entry][0][1])]
    
    return vocab

def saveVocab(vocab):
    
    '''
    The generated vocab file is similar to the one produced by the
    bert tokenizer module based on WordPiece tokenization.
    Fragments starting with ## represent those pieces of words that can
    just be used to complete another piece of word.
    '''
    
    ending_alphabet_chars = ["##"+char for char in alphabet_chars]
    ending_special_chars = ["##"+char for char in special_chars]
    
    bases = {word[0] for word in vocab if (not word[0] in alphabet_chars and not word[0] in special_chars)}
    rhymes = {word[1] for word in vocab if (not word[1] in ending_alphabet_chars and not word[1] in ending_special_chars and word[1]!="")}
    
    list_vocabs = []
    list_vocabs.extend(special_tokens)
    list_vocabs.extend(alphabet_chars)
    list_vocabs.extend(special_chars)
    list_vocabs.extend(bases)
    list_vocabs.extend(rhymes)
    list_vocabs.extend(ending_alphabet_chars)
    list_vocabs.extend(ending_special_chars)
    
    with open(vocab_file_name + ".txt", 'w',encoding="utf-8") as f:
        for token in list_vocabs:
            if token!="" and token!="\n":
                print(token, file=f)
                
def existsVocab():
    
    return os.path.isfile(vocab_file_name + ".txt")

def getVocab():
    
    if not existsVocab():
            runVocabCreation()
        
    vocab = Preprocessor.TextProcessor(vocab_file_name + ".txt")
    vocab.splitLines()

    return vocab.lines, special_tokens
            
def runVocabCreation():
    vocab = createVocab()
    saveVocab(vocab)