# -*- coding: utf-8 -*-

'''
This module creates the vocabulary file containing all the complete words from
Divina Commedia.
'''

import pickle
import os.path
import Preprocessor

pkl_file = "Dante/dantes_dictionary.pkl"
vocab_file_name = "./Vocab/FullVocab"

def openPKL(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data
    
def getWord(entry):
    
    list_ = entry[0]
    word = list_[1].replace("_","").replace("|","")
    
    return word
    
def createVocab():
    
    data = openPKL(pkl_file)
    
    vocab = set([getWord(data[entry]) for entry in data])
    
    return vocab

def saveVocab(vocab):
    
    list_vocabs = list(vocab)
    
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

    return vocab.lines
            
def runVocabCreation():
    vocab = createVocab()
    saveVocab(vocab)