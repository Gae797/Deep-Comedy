# -*- coding: utf-8 -*-

'''
This module contains methods to generate a dataset for the syllabification task,
both training and test set.
'''

import os.path

import Preprocessor

import tensorflow as tf
import tensorflow_text as text

punctuation = [",",".",";",":","!","?","»","«","“","”","(",")","-"]
digits = ["1","2","3","4","5","6","7","8","9","0"]

input_texts = ["Dante/inferno.txt","Dante/purgatorio.txt","Dante/paradiso.txt"]
output_texts = ["Dante/inferno_syllnew.txt","Dante/purgatorio_syllnew.txt","Dante/paradiso_syllnew.txt"]

input_file_name = "./Preprocessed data/SyllableTransformer/Syllabification_input"
output_file_name = "./Preprocessed data/SyllableTransformer/Syllabification_output"

def preprocess():
    
    '''
    This function creates one input and one output files for dataset generation
    based on the three full texts (Inferno, Purgatorio, Paradiso) and their
    syllabified version.
    '''
    
    inp=[]
    for input_ in input_texts:
        inp.extend(generateInput(input_,punctuation+digits))
        
    outp=[]
    for output_ in output_texts:
        outp.extend(generateOutput(output_,punctuation+digits))
        
    saveLines(inp,input_file_name)
    saveLines(outp,output_file_name)

def generateInput(path, char_remove_list):
    prep = Preprocessor.TextProcessor(path)
    prep.removeCharacters(char_remove_list)
    prep.lowerCase()
    prep.splitLines()
    prep.removeEmptyLines()
    prep.removePaddingStart()
    prep.addStartEndTokens()
    
    return prep.lines
    
def generateOutput(path, char_remove_list):
    prep = Preprocessor.TextProcessor(path)
    prep.removeCharacters(char_remove_list)
    prep.lowerCase()
    prep.splitLines()
    prep.removeEmptyLines()
    prep.removePaddingStart()
    prep.removePaddingEnd()
    prep.addStartEndTokens()

    return prep.lines

def saveLines(lines, title, train_count = 14133): #total lines = 14233
    
    if not lines:
        return
    
    f_training = open(title+"_training.txt", "a",encoding="utf8")
    f_test = open(title+"_test.txt", "a",encoding="utf8")
    
    counter = 0
    for line in lines:
        if counter<train_count:
            f_training.write(line+"\n")
        else:
            f_test.write(line+"\n")
            
        counter+=1
    
    f_training.close()
    f_test.close()
    
def loadLines(path):
    
    prep = Preprocessor.TextProcessor(path)
    prep.splitLines()
    
    return prep.lines

def getTokenizer():
    
    '''
    The tokenization for the syllabification task occurs on a char level; hence
    the unicode tokenizer is the chosen one.
    '''
    
    return text.UnicodeCharTokenizer()

#------------------------------------------------------------------------------

def outputGenerator(input_lines, output_lines):
    
    '''
    This function generates couple of input-target for the training session.
    Where the input is a preprocessed verse, while the output is the same verse
    having marked syllables.
    '''
    
    i = 0
    while i<len(input_lines):
        yield input_lines[i], output_lines[i]
        i+=1

def createDataset(input_path, output_path):
    
    if not os.path.isfile(input_path):
        preprocess()
    
    input_lines = loadLines(input_path)
    output_lines = loadLines(output_path)
    
    dataset = tf.data.Dataset.from_generator(outputGenerator, args=[input_lines,output_lines], 
                                             output_types=(tf.string,tf.string),
                                             output_shapes=((),()))
    
    return dataset