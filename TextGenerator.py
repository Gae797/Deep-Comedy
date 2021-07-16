# -*- coding: utf-8 -*-

'''
This module combines the two transformers in order to generate new lines made
up of hendacasyllables.
'''

import tensorflow as tf

import TransformerSyllable as tfs
import TransformerGenerator as tfg
from Transformer import create_masks

import SyllableDatasetGenerator as dg
import TextDatasetGenerator as dg2

BACKTRACK_DEPTH = 2
N_LINES = 33
REVERSE = dg2.REVERSE
FORCE_RHYME = True
ACCEPTABLE_THRESHOLD = 0.0

START_TOKEN = "S"
END_TOKEN = "E"

INPUT_LINES = ["SLA caron dimonio con occhi di bragia ELA",
               "SLB loro accennando tutte le raccoglie ELB",
               "SLC batte col remo qualunque s' adagia ELC"]

#------------------------------------------------------------------------------

generator = tfg.getTransformer()
syllabier = tfs.getTransformer()

gen_tokenizer = dg2.getTokenizer()
syll_tokenizer = dg.getTokenizer()

max_length = dg2.max_length

def syllabify(inp_sentence):

  tokenizer = syll_tokenizer
  transformer = syllabier
  
  inp_sentence = tf.convert_to_tensor([inp_sentence])
  inp_sentence = tokenizer.tokenize(inp_sentence).to_tensor()
  inp_sentence = tf.cast(inp_sentence,tf.int64)
  
  encoder_input = inp_sentence

  start = tokenizer.tokenize([START_TOKEN])[0]
  output = tf.convert_to_tensor([start])
  output = tf.cast(output, dtype=tf.int64)

  last_pred = START_TOKEN
  while last_pred!=END_TOKEN:
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    predictions, attention_weights = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    predictions = predictions[:, -1:, :]

    predicted_id = tf.argmax(predictions, axis=-1)
    last_pred = tokenizer.detokenize(tf.cast(predicted_id,dtype=tf.int32))

    output = tf.concat([output, predicted_id], axis=-1)

  output = tf.cast(output,dtype=tf.int32)
  text_ = tokenizer.detokenize(output)[0]
  text_ = text_.numpy().decode("utf-8")

  return text_

def countSyllables(sentence):
    
    syllabied_sentence = syllabify(sentence)
    return syllabied_sentence.count("|")

def isHendacasyllable(sentence):
    
    n = countSyllables(sentence)
    return n==11

def getLastLine(output):
    
    last_index_start = output.rindex(START_TOKEN)
    return output[last_index_start:]

def generateNextToken(encoder_input, partial_output, last_prediction, counter,results):
    
    tokenizer = gen_tokenizer
    transformer = generator
    
    end_char = START_TOKEN if REVERSE else END_TOKEN
    
    if counter==max_length or last_prediction[0]==end_char:
        if REVERSE:
            output = tokenizer.detokenizeReverse(partial_output)
            if output[-1]==" ":
                output = output[:-1]
        else:
            output = tokenizer.detokenize(partial_output)
        output = getLastLine(output)
        test_output = START_TOKEN + output[4:len(output)-4] + END_TOKEN
        if isHendacasyllable(test_output):
            #print(output)
            results.append(output)
            return True
        else:
            return False
    
    else:
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, partial_output)

        predictions, attention_weights = transformer(encoder_input,
                                                     partial_output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
    
        predictions = predictions[:, -1:, :]
    
        top_probs, top_indices = tf.nn.top_k(predictions,BACKTRACK_DEPTH)
        top_indices = tf.cast(top_indices,dtype=tf.int64)
        top_indices = top_indices.numpy()[0][0]
        top_probs = top_probs.numpy()[0][0]
        
        predicted_ids = []
        for i in top_indices:
            predicted_ids.append(tf.constant([[i]],dtype=tf.int64))
        
        for predicted_id, probability in zip(predicted_ids, top_probs):

            if probability<ACCEPTABLE_THRESHOLD:
                return False
            
            last_pred = tokenizer.detokenize(predicted_id)
            new_output = tf.concat([partial_output, predicted_id], axis=-1)
            valid_choice = generateNextToken(encoder_input, new_output, 
                                             last_pred, counter+1, results)
            
            if valid_choice:
                return True
            
            if not valid_choice and (last_pred[0]==end_char or last_pred==" "):
                return False
            
        return False
    
def generateNextLine(inp, lines):
    
    tokenizer = gen_tokenizer
    
    if REVERSE:
        inp_sentence = tokenizer.tokenizeReverse(inp, max_length)
    else:
        inp_sentence = tokenizer.tokenize(inp, max_length)
    inp_sentence = tf.expand_dims(inp_sentence,0)
      
    encoder_input = inp_sentence
    
    if REVERSE:
        start = tokenizer.tokenizeReverse(inp, max_length,padding=False)
        output = tf.convert_to_tensor([start])
        if FORCE_RHYME:
            rhyme_token = getRhymeToken(lines)
            if rhyme_token!=None:
                output = tf.concat([output, [rhyme_token]], axis=-1)
            
    else:
        start = tokenizer.tokenize(inp, max_length,padding=False)
        output = tf.convert_to_tensor([start])
    
    start_char = END_TOKEN if REVERSE else START_TOKEN
    success = generateNextToken(encoder_input,output,start_char,1,lines)
    
    return success

def getRhymeToken(lines):
    
    tokenizer = gen_tokenizer
    
    last_line = lines[-1]
    last_char = last_line[-1]
    
    if last_char=="A":
        rhyme_token=None
    else:
        target_line = lines[-2]
        target_words = target_line.split(" ")
        target_word = target_words[-2]
        
        if tokenizer.TYPE=="Syllable":
            tokenized_word = tokenizer.tokenize(target_word, max_length, padding=False, autoSpace=False)
        else:
            tokenized_word = tokenizer.tokenize(target_word, max_length, padding=False)
        
        rhyme_token = tokenized_word[-1]
        
        new_last_char = "A" if last_char=="C" else "C"
        start_tokens = tokenizer.tokenize("EL"+new_last_char,max_length,padding=False)
        rhyme_token = tf.concat([start_tokens, [rhyme_token]], axis=-1)
        
    return rhyme_token

def postProcessLine(line):
    
    new_line = line.replace("SLA ","").replace(" ELA","").replace("SLB ","").replace(" ELB","").replace("SLC ","").replace(" ELC","")
    new_line = new_line[0].upper() + new_line[1:] + "."
    
    return new_line

def postProcessLines(lines):
    
    new_lines = []
    counter=1
    for line in lines:
        new_lines.append(postProcessLine(line))
        if counter%3==0:
            new_lines.append("\n")
        counter+=1
        
    return new_lines

def generateLines(input_lines, n_lines=N_LINES):
    
    space = " "
    
    while len(input_lines)<n_lines:
        inp = input_lines[-3] + space + input_lines[-2] + space + input_lines[-1]
        generateNextLine(inp, input_lines)
        
    return postProcessLines(input_lines)

def generateText():
    generated_text = generateLines(INPUT_LINES)
    for line in generated_text:
        print(line)
        
    return generated_text, getSyllabifiedVersion(generated_text)
        
def getSyllabifiedVersion(lines):
    
    new_lines = []
    for line in lines:
        if line=="\n":
            new_lines.append("\n")
        else:
            line_ = line[:-1]
            line_ = line_.lower()
            input_ = START_TOKEN + line_ + END_TOKEN
            syll_version = syllabify(input_)
            new_lines.append(syll_version[1:-1])
            
    return new_lines
 
'''
text, syll_text = generateText()
for line in syll_text:
        print(line)
'''