# -*- coding: utf-8 -*-

'''
This module contains methods to generate the training dataset for the text
generation task. It also generates a file used for testing the generation of
new verses.
'''

import random
import os.path

import Preprocessor
import RhymeVocabGenerator as rvg
import SyllableVocabGenerator as svg
import FullVocabGenerator as fvg
from Tokenizer import RhymeTokenizer, SyllableTokenizer

import tensorflow as tf

punctuation = [",",".",";",":","!","?","»","«","“","”","(",")","-"]
digits = ["1","2","3","4","5","6","7","8","9","0"]

input_texts = ["Dante/inferno.txt","Dante/purgatorio.txt","Dante/paradiso.txt"]

input_file_name = "./Preprocessed data/TextGenerationTransformer/Tercet_input_training"
output_file_name = "./Preprocessed data/TextGenerationTransformer/Tercet_output_training"
test_file_name = "./Preprocessed data/TextGenerationTransformer/Generation_test"

TEST_NUMBER = 10 #Number of tercets to be used in the test session
RHYME_TOKENIZATION = True #False = Syllable tokenization
REVERSE = True

full_vocab = fvg.getVocab()

if RHYME_TOKENIZATION:
    vocab, reserved_tokens = rvg.getVocab()
    tokenizer = RhymeTokenizer(vocab, reserved_tokens)
    max_length = 90
    
else:
    vocab,reserved_tokens = svg.getVocab()
    tokenizer = SyllableTokenizer(vocab, reserved_tokens)
    max_length = 150

def preprocess():
    
    inp=[]
    for input_ in input_texts:
        inp.extend(generateInput(input_,punctuation+digits))
        
    outp=[]
    for output_ in input_texts:
        outp.extend(generateOutput(output_,punctuation+digits))
        
    test_indices = []
    while len(test_indices)<TEST_NUMBER*3:
        index = random.randrange(0,len(outp)-2,3)
        if index not in test_indices:
            test_indices.append(index)
            test_indices.append(index+1)
            test_indices.append(index+2)
        
    saveLines(inp,input_file_name,True, test_indices)
    saveLines(outp,output_file_name,False, test_indices)

def generateInput(path, char_remove_list):
    prep = Preprocessor.TextProcessor(path)
    prep.removeCharacters(char_remove_list)
    prep.lowerCase()
    prep.splitLines()
    prep.removeSingleLines()
    prep.removeEmptyLines()
    prep.removePaddingStart()
    prep.addSpecialTokens()
    
    return prep.lines
    
def generateOutput(path, char_remove_list):
    prep = Preprocessor.TextProcessor(path)
    prep.removeCharacters(char_remove_list)
    prep.lowerCase()
    prep.splitLines()
    prep.removeSingleLines()
    prep.removeEmptyLines()
    prep.removePaddingStart()
    prep.addSpecialTokens()
    prep.removeFirstLines(3)
    
    return prep.lines

def saveLines(lines, title, input_type, test_indices): #total lines = 14133
    
    if not lines:
        return
    
    f = open(title+".txt", "a",encoding="utf8")
    
    if input_type:
        f_test = open(test_file_name + ".txt","a",encoding="utf-8")
    
    for i in range(len(lines)):
        line = lines[i]
        if i in test_indices:
            if input_type:
                f_test.write(line + "\n")
                
        else:        
            f.write(line+"\n")
    
    f.close()
    
    if input_type:
        f_test.close()
    
def loadLines(path):
    
    prep = Preprocessor.TextProcessor(path)
    prep.splitLines()
    
    return prep.lines

#------------------------------------------------------------------------------
            
def getTokenizer():
    
    return tokenizer

def getVocabSize():
    
    return len(vocab)

#------------------------------------------------------------------------------

def outputGenerator(input_lines, output_lines):
    
    '''
    This generator produces input and output for the training session.
    The input is made up of three consecutive verses; the output consists of
    the same three verses plus the following one.
    '''
    
    i = 0
    space = str.encode(" ")
    while i<len(output_lines):
        inp = input_lines[i]+space+input_lines[i+1]+space+input_lines[i+2]
        out = inp + space + output_lines[i]
        inp = tokenizer.tokenize(inp,max_length)
        out = tokenizer.tokenize(out,max_length)
        yield inp,out 
        i+=1
        
def reverse_outputGenerator(input_lines, output_lines):
    
    '''
    This generator is identical to the previous one, but it works with reverse
    tokenization.
    '''
    
    i = 0
    space = str.encode(" ")
    while i<len(output_lines):
        inp = input_lines[i]+space+input_lines[i+1]+space+input_lines[i+2]
        out = inp + space + output_lines[i]
        inp = tokenizer.tokenizeReverse(inp,max_length)
        out = tokenizer.tokenizeReverse(out,max_length)
        
        yield inp,out 
        i+=1

def createDataset(input_path, output_path):
    
    if not os.path.isfile(input_path):
        preprocess()
    
    input_lines = loadLines(input_path)
    output_lines = loadLines(output_path)
    
    if REVERSE:
        dataset = tf.data.Dataset.from_generator(reverse_outputGenerator, args=[input_lines,output_lines], 
                                                 output_types=(tf.int64,tf.int64),
                                                 output_shapes=((max_length),(max_length)))
    else:
        dataset = tf.data.Dataset.from_generator(outputGenerator, args=[input_lines,output_lines], 
                                                 output_types=(tf.int64,tf.int64),
                                                 output_shapes=((max_length),(max_length)))
    
    return dataset