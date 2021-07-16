# -*- coding: utf-8 -*-

'''
This module contains classes of completely custom tokenizers.
'''

import tensorflow as tf

class RhymeTokenizer:
    
    '''
    This tokenizer assign two tokens for each word: one for the ending part
    responsible for the rhymes and one for the other part.
    In case of words where the rhyming part represents the whole word, a single
    token is produced.
    In case of unknown words (missing inside the vocabulary file), a simple char
    decomposition is applied (one token for each character).
    '''
    
    def __init__(self, vocab, reserved_tokens):
        
        self.vocab = vocab
        self.reserved_tokens = reserved_tokens
        
        self.TYPE = "Rhyme"
        
    def charDecomposition(self,word):
        
        '''
        This method decompose each word or piece of word into the corresponding
        characters.
        '''
        
        chars = []
        chars.append(word[0])
        for char in word:
            chars.append("##"+char)
            
        indexes = [self.findIndex(char) for char in chars]
        return indexes
        
    def findIndex(self, word):
        
        for i in range(len(self.vocab)):
            if self.vocab[i]==word:
                return i
                
        return None
    
    def getStringByIndex(self, index):
        
        for i in range(len(self.vocab)):
            if i==index:
                return self.vocab[i]
            
        return "[UKN]"
        
    def getLongestEnd(self, word):
        
        for i in range(0,len(word)):
            subword = word[i:]
            target = "##" + subword
            token = self.findIndex(target)
            
            if token!=None:
                return token, i
            
        return None,-1
            
    def tokenizeSentence(self, sentence):
        
        '''
        The tokenization of each word consists of dividing it into the rhyme
        component and the other part by searching for the longest rhyme
        correspondence (##).
        '''
            
        if not isinstance(sentence, str):
            sentence = sentence.decode("utf-8")
            
        words = sentence.split(" ")
    
        tokenized_words = []
        
        for _word in words:
            
            if _word!="":
                
                word = _word[0:]
                
                if word in self.reserved_tokens:
                    tokenized_words.append([self.findIndex(word)])
                else:
                    
                    if word in self.vocab:
                        end_token = None
                    
                    else:
                        end_token, end_index = self.getLongestEnd(word)
                    
                        if end_index==0:
                            end_token = None
                    
                    if end_token==None:
                        start_word = word[0:]
                    else:
                        start_word = word[:end_index]
                    start_token = self.findIndex(start_word)
                    
                    if start_token==None:
                        tokenized_words.append(self.charDecomposition(word))
                        
                    else:
                        if end_token==None:
                            tokenized_words.append([start_token])
                        else:
                            tokenized_words.append([start_token,end_token])
            
        return tokenized_words
        
    def tokenize(self, inp, max_length, padding=True):

        tokenized_inp = self.tokenizeSentence(inp)
        
        tokenized_tensor = tf.ragged.constant([tokenized_inp])
        tokenized_tensor = tokenized_tensor.merge_dims(-2,-1)
        tokenized_tensor = tokenized_tensor.to_tensor()
        
        if padding:
            paddings = [[0, 0], [0, max_length-tf.shape(tokenized_tensor[0])[0]]]
            tokenized_tensor = tf.pad(tokenized_tensor, paddings, "CONSTANT")
        
        tokenized_tensor = tf.cast(tokenized_tensor,tf.int64)
        
        return tokenized_tensor[0]
    
    def detokenize(self, inp):
        
        tokens = inp.numpy()[0]
        words = [self.getStringByIndex(token) for token in tokens]
        
        detokenized_sentence = ""
        for word in words:
            if word.startswith("##"):
                end_word = word[2:]
            
                detokenized_sentence+=end_word
            else:
                detokenized_sentence+=" " + word
        
        if detokenized_sentence[0]==" " and len(detokenized_sentence)!=1:
            detokenized_sentence = detokenized_sentence[1:]
            
        detokenized_sentence = detokenized_sentence.replace(" '","'")
        
        return detokenized_sentence
    
    def divideToVerses(self, inp):
        
        verses=[]
        if not isinstance(inp, str):
            inp = inp.decode("utf-8")
            
        index=inp.find("E")
        while index!=-1:
            end_index = index+3
            verse = inp[:end_index]
            verses.append(verse)
            
            if end_index==len(inp):
                break
            
            inp = inp[end_index:]
            index = inp.find("E")
        
        return verses
    
    def tokenizeReverse(self, inp, max_length, padding=True):
        
        verses = self.divideToVerses(inp)
        tokenized_inps = []
        for verse in verses:
            tokenized_verse = self.tokenize(verse, max_length, padding=False)
            tokenized_verse = tokenized_verse.numpy()
            tokenized_verse = tokenized_verse[::-1]
            tokenized_verse = tf.constant(tokenized_verse)
            tokenized_inps.append(tokenized_verse)
            
        tokenized_inp = tokenized_inps[0]
        for i in range(1,len(tokenized_inps)):
            tokenized_inp = tf.concat([tokenized_inp, tokenized_inps[i]], axis=-1)
        
        tokenized_inp = tf.expand_dims(tokenized_inp,0)
        if padding:
            paddings = [[0, 0], [0, max_length-tf.shape(tokenized_inp[0])[0]]]
            tokenized_inp = tf.pad(tokenized_inp, paddings, "CONSTANT")
            
        return tokenized_inp[0]
    
    def detokenizeReverse(self, inp):
        
        '''
        Each verse is tokenized and then the resulting tensor of tokens is
        reversed.
        '''
        
        tokens = inp.numpy()[0]
        words_ = [self.getStringByIndex(token) for token in tokens]
        
        indices = [0]
        for i in range(len(words_)):
            if words_[i].startswith("SL"):
                indices.append(i+1)
                
        sentences = []
        for value in range(len(indices)-1):
            i1 = indices[value]
            i2 = indices[value+1]
            new_sentence = words_[i1:i2]
            new_sentence.reverse()
            sentences.append(new_sentence)
            
        elaborated_sentences = []
        for words in sentences:
            detokenized_sentence = ""
            for word in words:
                if word.startswith("##"):
                    end_word = word[2:]
                
                    detokenized_sentence+=end_word
                else:
                    detokenized_sentence+=" " + word
            
            if detokenized_sentence[0]==" " and len(detokenized_sentence)!=1:
                detokenized_sentence = detokenized_sentence[1:]
                
            detokenized_sentence = detokenized_sentence.replace(" '","'")
            elaborated_sentences.append(detokenized_sentence)
            
        result = ""
        for elaborated_sentence in elaborated_sentences:
            result+= elaborated_sentence
        
        return result
        
#------------------------------------------------------------------------------
    
class SyllableTokenizer:
    
    '''
    This tokenizer assign a token to each syllable belonging to the assigned
    vocabulary file.
    '''
    
    def __init__(self, vocab, reserved_tokens):
        
        self.vocab = vocab
        self.reserved_tokens = reserved_tokens
        
        self.TYPE = "Syllable"
        
    def findIndex(self, word):
        
        for i in range(len(self.vocab)):
            if self.vocab[i]==word:
                return i
                
        return None
    
    def getStringByIndex(self, index):
        
        for i in range(len(self.vocab)):
            if i==index:
                return self.vocab[i]
            
        return "[UKN]"
    
    def getLongestStart(self, word):
        
        for i in range(len(word)-1,-1,-1):
            if i==0:
                target = word[0]
            else:
                target = word[:i+1]
            token = self.findIndex(target)
            
            if token!=None:
                return token, i
            
        return None,-1
            
    def tokenizeSentence(self, sentence, autoSpace):
            
        if not isinstance(sentence, str):
            sentence = sentence.decode("utf-8")
            
        words = sentence.split(" ")
    
        tokenized_words = []
        
        for _word in words:
            
            if _word!="":
                
                word = _word[0:]
                
                if word in self.reserved_tokens:
                    if autoSpace:
                        tokenized_words.append([self.findIndex(word),self.findIndex("SPACE")])
                    else:
                        tokenized_words.append([self.findIndex(word)])
                else:
                    tokens = []
                    syll_token, index = self.getLongestStart(word)
                    
                    if syll_token==None:
                            raise "Missing token for: "+word
                    
                    tokens.append(syll_token)
                    while index!=len(word)-1:
                        word = word[index+1:]
                        syll_token, index = self.getLongestStart(word)
                        
                        if syll_token==None:
                            raise "Missing token for: "+word
                        
                        tokens.append(syll_token)
                    
                    if autoSpace:
                        tokens.append(self.findIndex("SPACE"))
                        
                    tokenized_words.append(tokens)
                        
        return tokenized_words
        
    def tokenize(self, inp, max_length, padding=True, autoSpace=True):

        tokenized_inp = self.tokenizeSentence(inp, autoSpace)
        
        tokenized_tensor = tf.ragged.constant([tokenized_inp])
        tokenized_tensor = tokenized_tensor.merge_dims(-2,-1)
        tokenized_tensor = tokenized_tensor.to_tensor()
        
        if padding:
            paddings = [[0, 0], [0, max_length-tf.shape(tokenized_tensor[0])[0]]]
            tokenized_tensor = tf.pad(tokenized_tensor, paddings, "CONSTANT")
        
        tokenized_tensor = tf.cast(tokenized_tensor,tf.int64)
        
        return tokenized_tensor[0]
    
    def detokenize(self, inp):
        
        tokens = inp.numpy()[0]
        words = [self.getStringByIndex(token) for token in tokens]
        
        detokenized_sentence = ""
        for word in words:
            if word=="SPACE":
                detokenized_sentence+=" "
            else:
                detokenized_sentence+=word
        
        if detokenized_sentence[0]==" " and len(detokenized_sentence)!=1:
            detokenized_sentence = detokenized_sentence[1:]
            
        detokenized_sentence = detokenized_sentence.replace(" '","'")
        
        return detokenized_sentence
    
    def divideToVerses(self,inp):
        
        verses=[]
        if not isinstance(inp, str):
            inp = inp.decode("utf-8")
            
        index=inp.find("E")
        while index!=-1:
            end_index = index+3
            verse = inp[:end_index]
            verses.append(verse)
            
            if end_index==len(inp):
                break
            
            inp = inp[end_index:]
            index = inp.find("E")
        
        return verses
    
    def tokenizeReverse(self, inp, max_length, padding=True):
        
        '''
        Each verse is tokenized and then the resulting tensor of tokens is
        reversed.
        '''
        
        verses = self.divideToVerses(inp)
        tokenized_inps = []
        for verse in verses:
            tokenized_verse = self.tokenize(verse, max_length, padding=False)
            tokenized_verse = tokenized_verse.numpy()
            tokenized_verse = tokenized_verse[::-1]
            tokenized_verse = tf.constant(tokenized_verse)
            tokenized_inps.append(tokenized_verse)
            
        tokenized_inp = tokenized_inps[0]
        for i in range(1,len(tokenized_inps)):
            tokenized_inp = tf.concat([tokenized_inp, tokenized_inps[i]], axis=-1)
        
        tokenized_inp = tf.expand_dims(tokenized_inp,0)
        
        if padding:
            paddings = [[0, 0], [0, max_length-tf.shape(tokenized_inp[0])[0]]]
            tokenized_inp = tf.pad(tokenized_inp, paddings, "CONSTANT")
        
            
        return tokenized_inp[0]
    
    def detokenizeReverse(self, inp):
        
        tokens = inp.numpy()[0]
        words_ = [self.getStringByIndex(token) for token in tokens]
        
        indices = [0]
        for i in range(len(words_)):
            if words_[i].startswith("SL"):
                indices.append(i+1)
                
        sentences = []
        for value in range(len(indices)-1):
            i1 = indices[value]
            i2 = indices[value+1]
            new_sentence = words_[i1:i2]
            new_sentence.reverse()
            sentences.append(new_sentence)
            
        elaborated_sentences = []
        for words in sentences:
            detokenized_sentence = ""
            for word in words:
                if word=="SPACE":
                    detokenized_sentence+=" "
                else:
                    detokenized_sentence+=word
            
            if detokenized_sentence[0]==" " and len(detokenized_sentence)!=1:
                detokenized_sentence = detokenized_sentence[1:]
                
            detokenized_sentence = detokenized_sentence.replace(" '","'")
            elaborated_sentences.append(detokenized_sentence)
            
        result = ""
        for elaborated_sentence in elaborated_sentences:
            result+= elaborated_sentence + " "
        result = result[:-1]
        
        return result