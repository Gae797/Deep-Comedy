# -*- coding: utf-8 -*-

'''
This module is responsible for evaluating the quality of the produced text.
'''

import TextDatasetGenerator

import RhymeVocabGenerator
from Tokenizer import RhymeTokenizer

import TextGenerator

N_LINES = 6

def wordCorrectness(line, vocab):
    
    line_ = line.replace(".","")
    words = line_.split(" ")
    counter=0
    for word in words:
        if word in vocab:
            counter+=1
            
    return counter/len(words)

def correctnessScore(lines):
    
    full_vocab = TextDatasetGenerator.full_vocab
    new_lines = lines[3:]
    
    scores = [wordCorrectness(line, full_vocab) for line in new_lines]
    
    return sum(scores) / len(scores)

def rhymeScore(lines):
    
    vocab, reserved_tokens = RhymeVocabGenerator.getVocab()
    tokenizer = RhymeTokenizer(vocab, reserved_tokens)
    
    total = 0
    correct = 0
    for i in range(len(lines)):
        if i>2:
            if i%3==0 or i%3==2:
                
                total+=1
                
                line = lines[i]
                line = line.replace(".","")
                last_word1 = line.split(" ")[-1]
                
                line = lines[i-2]
                line = line.replace(".","")
                last_word2 = line.split(" ")[-1]
                
                tokens = tokenizer.tokenize(last_word1, TextDatasetGenerator.max_length, padding=False)
                last_token_1 = tokens.numpy()[-1]
                
                tokens = tokenizer.tokenize(last_word2, TextDatasetGenerator.max_length, padding=False)
                last_token_2 = tokens.numpy()[-1]
                
                if last_token_1==last_token_2:
                    correct+=1
                    
    return correct/total

def varietyScore(lines):
    
    scores = []
    for i in range(3,len(lines),3):
        
        line = lines[i] + " " + lines[i+1] + " " + lines[i+2]
        words = line.split(" ")
        words = [word for word in words if len(word)>3]
        non_repeating_words = set(words)
        
        difference = len(words) - len (non_repeating_words)
        score = 1 - difference/len(words)
        scores.append(score)
        
    return sum(scores) / len(scores)

def getScore(lines):
    
    score1 = correctnessScore(lines)
    score2 = rhymeScore(lines)
    score3 = varietyScore(lines)
    
    final_score = (score1*2 + score2*2 + score3) / 5
    
    return (score1, score2, score3, final_score)

def runTest(tercet):
    
    new_lines = TextGenerator.generateLines(tercet, n_lines=N_LINES)
    new_lines = [line for line in new_lines if line!="\n"]
    
    return getScore(new_lines)

def runFullTest():
    
    lines = TextDatasetGenerator.loadLines(TextDatasetGenerator.test_file_name + ".txt")
    
    tercets = []
    for i in range(len(lines)):
        if i%3==0:
            tercet = []
        tercet.append(lines[i])
        if i%3==2:
            tercets.append(tercet)
    
    scores = []
    i=1
    for tercet in tercets:
        print("Running test {}/{}...".format(i,len(tercets)))
        scores.append(runTest(tercet))
        i+=1
    
    scores1 = []
    scores2 = []
    scores3 = []
    final_scores = []
    for score in scores:
        score1, score2, score3, final_score = score
        scores1.append(score1)
        scores2.append(score2)
        scores3.append(score3)
        final_scores.append(final_score)
        
    score = {
        "Word correctness": mean(scores1),
        "Rhyme correctness": mean(scores2),
        "Word variety": mean(scores3),
        "Score": mean(final_scores)
        }
    
    return score

def mean(list_):
    
    return sum(list_)/len(list_)

#runFullTest()