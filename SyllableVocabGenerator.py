# -*- coding: utf-8 -*-

'''
This module creates the vocabulary file used by the SyllableTokenizer.
'''

import pickle
import os.path
import Preprocessor

pkl_file = "Dante/dantes_dictionary.pkl"
vocab_file_name = "./Vocab/SyllableVocab"

special_tokens = ["PAD","SLA","ELA","SLB","ELB","SLC","ELC","SPACE"]
alphabet_chars = ["a","b","c","d","e","f","g","h","i","j","l",
                  "m","n","o","p","q","r","s","t","u","v","x","y","z","'"]
vowels=["a","e","i","o","u","à","ä","è","é","ë","ì","ï","ò","ó","ö","ù","ü"]

def openPKL(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data
    
def getSyllables(entry, syllables):
    
    list_ = entry[0]
    syllb_word = list_[1].replace("_","")
    
    new_syll = ""
    for c in syllb_word:
        if c=="|":
            syllables.add(new_syll)
            new_syll=""
        else:
            new_syll+=c
    syllables.add(new_syll)
    
def createVocab():
    
    data = openPKL(pkl_file)
    
    vocab = set()
    vocab.update(alphabet_chars)
    vocab.update(vowels)
    
    for entry in data:
        getSyllables(data[entry], vocab)
    
    return vocab

def saveVocab(vocab):
    
    list_vocabs = []
    list_vocabs.extend(special_tokens)
    list_vocabs.extend(vocab)
    
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