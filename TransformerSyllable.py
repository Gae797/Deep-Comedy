# -*- coding: utf-8 -*-

'''
This is the implementation of a trasformer applied to the syllabification task.
It can be used both for training or testing the model.
'''

import logging

import tensorflow_text as text
import tensorflow as tf

import SyllableDatasetGenerator

from Transformer import *

logging.getLogger('tensorflow').setLevel(logging.ERROR)

input_file_name = SyllableDatasetGenerator.input_file_name
output_file_name = SyllableDatasetGenerator.output_file_name
checkpoint_path = "./Checkpoints/SyllableTransformer"

BUFFER_SIZE = 20000
BATCH_SIZE = 64

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

EPOCHS = 50

VERBOSE = True

input_vocab_size  = 253 #253 is maximum unicode value (same as token value) inside the training dataset
output_vocab_size = 253

START_TOKEN = "S"
END_TOKEN = "E"

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer=None
optimizer = None
train_batches = None
ckpt_manager = None

#-----------------------------------------------------------------------------

def createTransformer():
    
    global transformer, optimizer, train_batches, ckpt_manager
    
    train_examples = SyllableDatasetGenerator.createDataset(input_file_name+"_training.txt",output_file_name+"_training.txt")

    train_batches = make_batches(train_examples)
    
    learning_rate = CustomSchedule(d_model)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=output_vocab_size,
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate)
    
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      if VERBOSE:
          print('Latest checkpoint restored for Syllable')

def trainTransformer():
    
    if transformer==None:
        createTransformer()
        
    f = open(checkpoint_path+"/Parameters.txt","w")
    f.write("Number of layers: {}\n".format(num_layers))
    f.write("d_model: {}\n".format(d_model))
    f.write("dff: {}\n".format(dff))
    f.write("Number of heads: {}\n".format(num_heads))
    f.write("Dropout rate: {}".format(dropout_rate))
    f.close()
        
    train(transformer, optimizer, train_loss, train_accuracy, 
          ckpt_manager, EPOCHS, train_batches, VERBOSE)

def testTransformer():
    
    if transformer==None:
        createTransformer()
        
    score = test()
    print(score)

def getTransformer():
    
    if transformer==None:
        createTransformer()
        
    return transformer

def tokenize_pairs(inp, out):
    
    tokenizer = text.UnicodeCharTokenizer()
    
    inp = tokenizer.tokenize(inp)
    inp = inp.to_tensor()
    inp = tf.cast(inp,tf.int64)

    out = tokenizer.tokenize(out)
    out = out.to_tensor()
    out = tf.cast(out,tf.int64)
    
    return inp, out

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))
  
#-----------------------------------------------------------------------------
#Evaluate and Test

def evaluate(inp_sentence):

  tokenizer = text.UnicodeCharTokenizer()
  
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

  return text_

def lineScore(inp, out):
    n_inp = inp.count("|")
    n_out = out.count("|")
    
    difference = abs(n_inp-n_out)
    score = 1 - difference/11
    
    return score

def test():
    
    inp_lines = SyllableDatasetGenerator.loadLines(input_file_name + "_test.txt")
    out_lines = SyllableDatasetGenerator.loadLines(output_file_name + "_test.txt")
    
    scores = []
    for inp, out in zip(inp_lines, out_lines):
        syll_inp = evaluate(inp)
        syll_inp = syll_inp.numpy().decode("utf-8")
        scores.append(lineScore(syll_inp, out))
        
    return sum(scores) / len(scores)