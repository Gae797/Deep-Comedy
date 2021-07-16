# -*- coding: utf-8 -*-

'''
This is the implementation of a trasformer applied to the text generation task.
It can be used both for training or testing the model.
'''

import logging

import tensorflow as tf

import TextDatasetGenerator

from Transformer import *

logging.getLogger('tensorflow').setLevel(logging.ERROR)

input_file_name = TextDatasetGenerator.input_file_name
output_file_name = TextDatasetGenerator.output_file_name
checkpoint_path = "./Checkpoints/RhymeReverse"

BATCH_SIZE = 64

num_layers = 4
d_model = 128
dff = 256
num_heads = 4
dropout_rate = 0.1

EPOCHS = 100

VERBOSE = True

input_vocab_size  = TextDatasetGenerator.getVocabSize()+1
output_vocab_size = TextDatasetGenerator.getVocabSize()+1

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer=None
optimizer = None
train_batches = None
ckpt_manager = None

#------------------------------------------------------------------------------

def createTransformer():
    
    global transformer, optimizer, train_batches, ckpt_manager
    
    train_examples = TextDatasetGenerator.createDataset(input_file_name+".txt",output_file_name+".txt")

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
          print('Latest checkpoint restored for Generator')

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
          ckpt_manager, EPOCHS, train_batches, VERBOSE, makePrediction)

def testTransformer():
    
    if transformer==None:
        createTransformer()
    
    makePrediction()
    #Use TextQuality.runFullTest to get the metrics score

def getTransformer():
    
    if transformer==None:
        createTransformer()
        
    return transformer

def make_batches(ds):
  return (
      ds
      .cache()
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE))

#------------------------------------------------------------------------------
#Evaluate and Test

def evaluate(trasformer, inp_sentence):

  tokenizer = DatasetGenerator2.getTokenizer()

  inp_sentence = tokenizer.tokenizeReverse(inp_sentence, TextDatasetGenerator.max_length)
  inp_sentence = tf.expand_dims(inp_sentence,0)

  encoder_input = inp_sentence

  if TextDatasetGenerator.REVERSE:
      start = tokenizer.tokenize("ELA", TextDatasetGenerator.max_length,padding=False)
  else:
      start = tokenizer.tokenize("SLA", TextDatasetGenerator.max_length,padding=False)
  output = tf.convert_to_tensor([start])

  last_pred = "ELA" if TextDatasetGenerator.REVERSE else "SLA"
  ending_pred = "SL" if TextDatasetGenerator.REVERSE else "EL"
  counter=0
  end_token_count = 0
  while counter<DatasetGenerator2.max_length:
     
    if last_pred.startswith(ending_pred):
        end_token_count+=1
        if end_token_count==4:
            break
      
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
    last_pred = tokenizer.detokenize(predicted_id)
    counter+=1

    output = tf.concat([output, predicted_id], axis=-1)
  
  text_ = tokenizer.detokenizeReverse(output)

  return text_

def makePrediction():
    
    test_sentence = "SLA nel mezzo del cammin di nostra vita ELA SLB mi ritrovai per una selva oscura ELB SLC ché la diritta via era smarrita ELC"
    prediction = evaluate(transformer,test_sentence)
    print("Predicted: \t"+prediction)