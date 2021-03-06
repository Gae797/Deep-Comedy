# The Deep Comedy 2.0

The aim of the project is to generate verses following the Divine Comedy's style.
In particular, each verse should follow the rhyme's scheme and it should be hendecasyllabic. The generation of new poetry is based on Deep Learning's techniques and particularly on the **Transformer** architecture, whose usages are twofold:

- A first neural network (**Syllabifier**) is responsible for translating each given verse in its correspondent syllabified version;
- The second transformer (**Generator**) is used to generate a new line, given the previous three ones (basically allowing infinite lines generation).

The two architectures are combined by means of an algorithm which lets the Generator produce tokens following a depth-first greedy search until the Syllabifier is able to divide its output into exactly 11 syllables.

The project contains four pre-trained networks, depending on the tokenization system (based on rhymes or syllables) and on the tokenization method (simple or reverse) for the Generator. The best results with respect to rhymes are obtained using a rhyme tokenizer in reverse mode, which consists of a normal tokenization where the order of tokens is reversed for each verse (making easier to predict rhymes).

### Files and Folders

- *Preprocessor.py* contains all the functions to open, edit and save text files for preprocessing;
- *RhymeVocabGenerator.py*, *SyllableVocabGenerator.py* and *FullVocabGenerator.py* are responsible for producing all the vocab files essential for text tokenization;
- *Tokenizer.py* includes the two customized tokenizers (RhymeTokenizer and SyllableTokenizer);
- *SyllableDatasetGenerator.py* and *TextDatasetGenerator.py* are modules that handle the generation of input/output datasets (for training and test purposes) for both the Syllabifier and the Generator;
- *Transformer.py* is the implementation of Transformer's architecture for neural networks using **Tensorflow**;
- *TransformerSyllable.py* and *TransformerGenerator.py* are the implementation of the Transformer for the Syllabifier and Generator respectively;
- *TextGenerator.py* contains all the methods to produce new verses;
- *TextQuality.py* is the module used to test the quality of the generated text;
- *Checkpoints* is a folder containing the saved weights for the trained models;
- *Dante* is a folder containing the full Divine Comedy, also in the syllabified version, and a dictionary of syllabified words covering the whole vocabulary. These files come from the following repository: https://github.com/asperti/Dante
- *Generated texts* contains examples of poetry produced by the pre-trained models;
- *Preprocessed data* is the target folder for all the files generated by dataset generators for training or testing;
- *Vocab* is the target folder for the vocab files produced by the vocab generators.

### How to generate new verses

Access the *TextDatasetGenerator.py* file and set a value ("True" or "False") for both "RHYME_TOKENIZATION" and REVERSE; then choose the respective file name (with respect to folders'names inside the *Checkpoint* directory) inside the variable "checkpointpath" of *TransformerGenerator.py*. Then it is enough to go inside the *TextGenerator.py* file, set values for "INPUT_LINES" and "N_LINES" and finally call the "generateText" function.

### How to train a new model

To train the Syllabifier or the Generator, just set the initial variables of the *TransformerSyllable.py* or *TransformerGenerator.py* files respectively to the desired values and call the function named "trainTransformer".

### How to test a model

In order to test a Syllabifier model, go inside the *TransformerSyllable.py* module and call the "testTransformer" function. In order to test the Generator model, go inside the *TextQuality.py* module and call the "runFullTest" function.