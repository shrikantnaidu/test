---
layout: post
title: Generate TV Scripts
date: 2020-06-08T 20:46:10 +03:00
description: "Building a recurrent neural network (RNN) to generate original TV scripts by training on Seinfeld dialogue data"
image: "assets/images/masonary-post/tv-scripts.jpg"
categories: 
  - "Deep Learning"
---

## [Generate TV Scripts](https://github.com/shrikantnaidu/Generate-TV-Scripts)

---
### Why We're Here 

In this project, we'll generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs. We'll be using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.  The Neural Network we'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

### Get the Data

The data resides in `./data/Seinfeld_Scripts.txt` and you're encouraged to open that file and look at the text. 
* a. As a first step, we'll load in this data and look at some samples. 
* b. Then, we'll define and train an RNN to generate a new script!

>```python
># load in data
>import helper
>
>data_dir = './data/Seinfeld_Scripts.txt'
>text = helper.load_data(data_dir)
>```

### Explore the Data

We play around with `view_line_range` to view different parts of the data. This will give us a sense of the data. For example, that it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`.


>```python
>view_line_range = (0, 10)
>
>import numpy as np
>
>print('Dataset Stats')
>print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
>
>lines = text.split('\n')
>print('Number of lines: {}'.format(len(lines)))
>word_count_line = [len(line.split()) for line in lines]
>print('Average number of words in each line: {}'.format(np.average(word_count_line)))
>
>print()
>print('The lines {} to {}:'.format(*view_line_range))
>print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
>```

```
Output:
    Dataset Stats
    Roughly the number of unique words: 46367
    Number of lines: 109233
    Average number of words in each line: 5.544240293684143
    
    The lines 0 to 10:
    jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 
    
    jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 
    
    george: are you through? 
    
    jerry: you do of course try on, when you buy? 
    
    george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
```

---
### Implement Pre-processing Functions

The first thing to do to any dataset is pre-processing. Implement the following pre-processing functions below:
- a. Lookup Table
- b. Tokenize Punctuation

#### Lookup Table

To create a word embedding, you first need to transform the words to ids. In this function, create two dictionaries:
- a. Dictionary to go from the words to an id, we'll call `vocab_to_int`
- b. Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following **tuple** `(vocab_to_int, int_to_vocab)`

>```python
>import problem_unittests as tests
>from collections import Counter
>
>def create_lookup_tables(text):
>    """
>    Create lookup tables for vocabulary
>    :param text: The text of tv scripts split into words
>    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
>    """
>
>    word_counts = Counter(text)
>    
>    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
>    
>    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
>    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
>    
>    return (vocab_to_int,int_to_vocab)
>
>
>tests.test_create_lookup_tables(create_lookup_tables)
>```

```
Output:
    Tests Passed
```

---
### Tokenize Punctuation

We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. 

We create a dictionary for the following symbols where the symbol is the key and value is the token:

- a. Period ( **.** )
- b. Comma ( **,** )
- c. Quotation Mark ( **"** )
- d. Semicolon ( **;** )
- e. Exclamation mark ( **!** )
- f. Question mark ( **?** )
- g. Left Parentheses ( **(** )
- h. Right Parentheses ( **)** )
- i. Dash ( **-** )
- j. Return ( **\n** )

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word.

>```python
>def token_lookup():
>    """
>    Generate a dict to turn punctuation into a token.
>    :return: Tokenized dictionary where the key is the punctuation and the value is the token
>    """
>    tokens = {
>                '.': '||period||',
>                ',': '||comma||',
>                '"': '||quotation_mark||',
>                ';': '||semicolon||',
>                '!': '||exclamation_mark||',
>                '?': '||question_mark||',
>                '(': '||left_parentheses||',
>                ')': '||right_Parentheses||',
>                '-': '||dash||',
>                '\n': '||return||'
>            }
>    
>    return tokens
>
>tests.test_tokenize(token_lookup)
>```

```
Output:
    Tests Passed
```

---
### Pre-process all the data and save it

We pre-process all the data and save it to a file.

>```python
>
># pre-process training data
>helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
>```

#### Check Point

This is the first checkpoint. If we have to restart the notebook, we can start from here. The preprocessed data has been saved to disk.


>```python
>
>import helper
>import problem_unittests as tests
>
>int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
>```

### Build the Neural Network

In this section, we'll build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

#### Check Access to GPU


>```python
>import torch
>
># Check for a GPU
>train_on_gpu = torch.cuda.is_available()
>if not train_on_gpu:
>    print('No GPU found. Please use a GPU to train your neural network.')
>```

### Input

Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

We can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.

>```python
>data = TensorDataset(feature_tensors, target_tensors)
>data_loader = torch.utils.data.DataLoader(data, 
>                                          batch_size=batch_size)
>```

### Batching

Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

We can batch words using the DataLoader, but it will be up to us to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.

For example, say we have these as input:

>```
>words = [1, 2, 3, 4, 5, 6, 7]
>sequence_length = 4
>```

Our first `feature_tensor` should contain the values:
>```
>[1, 2, 3, 4]
>```

And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
>```
>5
>```

This should continue with the second `feature_tensor`, `target_tensor` being:
>```
>[2, 3, 4, 5]  # features
>6             # target
>```

Code Snippet:

>```python
>from torch.utils.data import TensorDataset, DataLoader
>import numpy as np
>
>
>def batch_data(words, sequence_length, batch_size):
>    """
>    Batch the neural network data using DataLoader
>    :param words: The word ids of the TV scripts
>    :param sequence_length: The sequence length of each batch
>    :param batch_size: The size of each batch; the number of sequences in a batch
>    :return: DataLoader with batched data
>    """
>
>    #number of batches
>    number_batches = len(words)//batch_size     
>    
>    # only take full batches
>    words = words[:number_batches*batch_size]
>        
>    # x -> feature , y -> target
>    x, y = [], []
>    
>    for ii in range(0, len(words)- sequence_length):
>        x.append(words[ii:ii+sequence_length])
>        y.append(words[ii + sequence_length])
>    
>       
>    #convert numpy arrays to tensors
>    x_tensors = torch.from_numpy(np.array(x))
>    y_tensors = torch.from_numpy(np.array(y))
>    
>    
>    #Dataset wrapping tensors
>    data = TensorDataset(x_tensors, y_tensors)
>    
>    #multi-process iterators over the dataset (our data loader)
>    data_loader = torch.utils.data.DataLoader(data, shuffle=True,
>                                          batch_size=batch_size)
>    
>    # return a dataloader
>    return data_loader
>       
>
># there is no test for this function, but you are encouraged to create
># print statements and tests of your own
>words = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
>sequence_length = 3
>data_loader = batch_data(words,sequence_length, 3)
>
>for feature, target in data_loader:
>    print("Feature: "+ str(feature))
>    print("Target: "+str(target))
>    print("Feature shape: "+str(feature.shape))
>    print("Target shape: "+str(target.shape))
>```

```
Output:
    Feature: tensor([[  3,   4,   5],
                     [  2,   3,   4],
                     [  9,  10,  11]])
    Target: tensor([  6,   5,  12])
    Feature shape: torch.Size([3, 3])
    Target shape: torch.Size([3])
    Feature: tensor([[  5,   6,   7],
                     [  8,   9,  10],
                     [  1,   2,   3]])
    Target: tensor([  8,  11,   4])
    Feature shape: torch.Size([3, 3])
    Target shape: torch.Size([3])
    Feature: tensor([[ 4,  5,  6],
                     [ 7,  8,  9],
                     [ 6,  7,  8]])
    Target: tensor([  7,  10,   9])
    Feature shape: torch.Size([3, 3])
    Target shape: torch.Size([3])
```

---
### Test your dataloader 

We'll have to modify this code to test a batching function, but it should look fairly similar.

Below, we're generating some test text data and defining a dataloader using the function you defined, above. Then, we are getting some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.

Our code should return something like the following (likely in a different order, if you shuffled your data):

```
torch.Size([10, 5])
tensor([[ 28,  29,  30,  31,  32],
        [ 21,  22,  23,  24,  25],
        [ 17,  18,  19,  20,  21],
        [ 34,  35,  36,  37,  38],
        [ 11,  12,  13,  14,  15],
        [ 23,  24,  25,  26,  27],
        [  6,   7,   8,   9,  10],
        [ 38,  39,  40,  41,  42],
        [ 25,  26,  27,  28,  29],
        [  7,   8,   9,  10,  11]])

torch.Size([10])
tensor([ 33,  26,  22,  39,  16,  28,  11,  43,  30,  12])
```

---
### Sizes

Our sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 

### Values

Notice that the targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.


>```python
># test dataloader
>
>test_text = range(50)
>t_loader = batch_data(test_text, sequence_length=5, batch_size=10)
>
>data_iter = iter(t_loader)
>sample_x, sample_y = data_iter.next()
>
>print(sample_x.shape)
>print(sample_x)
>print()
>print(sample_y.shape)
>print(sample_y)
>```

```
Output:
    torch.Size([10, 5])
    tensor([[ 26,  27,  28,  29,  30],
            [  3,   4,   5,   6,   7],
            [ 29,  30,  31,  32,  33],
            [ 28,  29,  30,  31,  32],
            [ 22,  23,  24,  25,  26],
            [ 44,  45,  46,  47,  48],
            [ 10,  11,  12,  13,  14],
            [ 25,  26,  27,  28,  29],
            [  4,   5,   6,   7,   8],
            [ 36,  37,  38,  39,  40]])
    
    torch.Size([10])
    tensor([ 31,   8,  34,  33,  27,  49,  15,  30,   9,  41])
```

---
### Build the Neural Network

We implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). You may choose to use a GRU or an LSTM. To complete the RNN, we'll have to implement the following functions for the class:
 - a. `__init__` - The initialize function. 
 - b. `init_hidden` - The initialization function for an LSTM/GRU hidden state
 - c. `forward` - Forward propagation function.
 
The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.

**The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

### Note

- a. Make sure to stack the outputs of the lstm to pass to your fully-connected layer, you can do this with `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
- b. You can get the last batch of word scores by shaping the output of the final, fully-connected layer like so:

    >```python
    ># reshape into (batch_size, seq_length, output_size)
    >output = output.view(batch_size, -1, self.output_size)
    ># get last batch
    >out = output[:, -1]
    >```

Now, we build the network keeping all the above things in mind.

>```python
>import torch.nn as nn
>
>class RNN(nn.Module):
>    
>    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
>        """
>        Initialize the PyTorch RNN Module
>        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
>        :param output_size: The number of output dimensions of the neural network
>        :param embedding_dim: The size of embeddings, should you choose to use them        
>        :param hidden_dim: The size of the hidden layer outputs
>        :param dropout: dropout to add in between LSTM/GRU layers
>        """
>        super(RNN, self).__init__()
>        
>        # set class variables
>        self.output_size = output_size
>        self.n_layers = n_layers
>        self.hidden_dim = hidden_dim
>        self.vocab_size = vocab_size
>        self.embedding_dim = embedding_dim
>        
>        self.dropout = nn.Dropout(0.20)
> 
>        # define model layers
>        # embedding and LSTM layers
>        self.embedding = nn.Embedding(vocab_size, embedding_dim)
>        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
>                            dropout = dropout, batch_first=True)
>        
>        
>        #linear fully connnected layer
>        self.fc = nn.Linear(hidden_dim, output_size)
>        
>        
>    def forward(self, nn_input, hidden):
>        """
>        Forward propagation of the neural network
>        :param nn_input: The input to the neural network
>        :param hidden: The hidden state        
>        :return: Two Tensors, the output of the neural network and the latest hidden state
>        """
>        
>        #first dimension is batch size
>        batch_size = nn_input.size(0)
>        
>        # embeddings and lstm_out
>        nn_input = nn_input.long()
>        embeds = self.embedding(nn_input)
>        lstm_out , hidden = self.lstm(embeds,hidden)
>        
>        
>        # stack up lstm outputs
>        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)
>        
>        # dropout and fully-connected layer
>        output = self.dropout(lstm_out)
>        output = self.fc(output)
>        
>        # reshape to be batch_size first
>        output = output.view(batch_size, -1, self.output_size)
>        out = output[:, -1] # get last batch of labels
>        
>        # return one batch of output word scores and the hidden state
>        return out, hidden
>        
>    
>    def init_hidden(self, batch_size):
>        '''
>        Initialize the hidden state of an LSTM/GRU
>        :param batch_size: The batch_size of the hidden state
>        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
>        '''
>        # Implement function
>        
>        # initialize hidden state with zero weights, and move to GPU if available
>        weight = next(self.parameters()).data
>        
>        if (train_on_gpu):
>            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
>                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
>        else:
>            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
>                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
>        
>        return hidden
>
>tests.test_rnn(RNN, train_on_gpu)
>```

```
Output:
    Tests Passed
```

---
### Define forward and backpropagation

Using the RNN class to apply forward and back propagation. This function will be called, iteratively, in the training loop as follows:

>```python
>loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
>```

And it should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.

**If a GPU is available, you should move your data to that GPU device, here.**


>```python
>def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
>    """
>    Forward and backward propagation on the neural network
>    :param decoder: The PyTorch Module that holds the neural network
>    :param decoder_optimizer: The PyTorch optimizer for the neural network
>    :param criterion: The PyTorch loss function
>    :param inp: A batch of input to the neural network
>    :param target: The target output for the batch of input
>    :return: The loss and the latest hidden state Tensor
>    """
>    
>    # move data to GPU, if available
>    if (train_on_gpu):
>        inp, target = inp.cuda(), target.cuda()
> 
>    # perform backpropagation and optimization
>    hidden = tuple([each.data for each in hidden])
>
>    # zero accumulated gradients
>    rnn.zero_grad()
>
>    # get the output from the model
>    output, hidden = rnn(inp, hidden)
>
>    # calculate the loss and perform backprop
>    loss = criterion(output, target)
>    loss.backward()
>    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
>    clip = 5
>    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
>    optimizer.step()    
>    # return the loss over a batch and the hidden state produced by our model
>    return loss.item(), hidden
>
># Note that these tests aren't completely extensive.
># they are here to act as general checks on the expected outputs of your functions
>
>tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
>```

```
Output:
    Tests Passed
``` 

---
### Neural Network Training

With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop

The training loop is implemented for you in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the `show_every_n_batches` parameter. You'll set this parameter along with other parameters in the next section.

>```python
> def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
>     batch_losses = []
>     
>     rnn.train()
> 
>     print("Training for %d epoch(s)..." % n_epochs)
>     for epoch_i in range(1, n_epochs + 1):
>         
>         # initialize hidden state
>         hidden = rnn.init_hidden(batch_size)
>         
>         for batch_i, (inputs, labels) in enumerate(train_loader, 1):
>             
>             # make sure you iterate over completely full batches, only
>             n_batches = len(train_loader.dataset)//batch_size
>             if(batch_i > n_batches):
>                 break
>             
>             # forward, back prop
>             loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
>             # record loss
>             batch_losses.append(loss)
> 
>             # printing loss stats
>             if batch_i % show_every_n_batches == 0:
>                 print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
>                     epoch_i, n_epochs, np.average(batch_losses)))
>                 batch_losses = []
> 
>     # returns a trained rnn
>     return rnn
>```

### Hyperparameters

Set and train the neural network with the following parameters:
- a. Set `sequence_length` to the length of a sequence.
- b. Set `batch_size` to the batch size.
- c. Set `num_epochs` to the number of epochs to train for.
- d. Set `learning_rate` to the learning rate for an Adam optimizer.
- e. Set `vocab_size` to the number of uniqe tokens in our vocabulary.
- f. Set `output_size` to the desired size of the output.
- g. Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
- h. Set `hidden_dim` to the hidden dimension of your RNN.
- i. Set `n_layers` to the number of layers/cells in your RNN.
- j. Set `show_every_n_batches` to the number of batches at which the neural network should print progress.

If the network isn't getting the desired results, tweak these parameters and/or the layers in the `RNN` class.

#### Data parameters:

###### a. Sequence Length
>sequence_length = 10  # of words in a sequence

###### b. Batch Size
>batch_size = 256

###### c. data loader - do not change
>train_loader = batch_data(int_text, sequence_length, batch_size)

#### Training parameters:
###### a. Number of Epochs
>num_epochs = 15

###### b. Learning Rate
>learning_rate = 0.001

#### Model parameters
###### a. Vocab size
>vocab_size = len(vocab_to_int)

###### b. Output size
>output_size = vocab_size

###### c. Embedding Dimension

>embedding_dim = 200

###### d. Hidden Dimension
>hidden_dim = 300

###### e. Number of RNN Layers
>n_layers = 2

###### f. Show stats for every n number of batches
>show_every_n_batches = 500

---
### Train

In the next cell, we'll train the neural network on the pre-processed data.  If you have a hard time getting a good loss, you may consider changing your hyperparameters. In general, we get better results with larger hidden and n_layer dimensions, but larger models take a longer time to train. 

**We are aiming for a loss less than 3.5.** 

We can also experiment with different sequence lengths, which determine the size of the long range dependencies that a model can learn.


>```python
>
># create model and move to gpu if available
>rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
>if train_on_gpu:
>    rnn.cuda()
>
># defining loss and optimization functions for training
>optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
>criterion = nn.CrossEntropyLoss()
>
># training the model
>trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)
>
># saving the trained model
>helper.save_model('./save/trained_rnn', trained_rnn)
>print('Model Trained and Saved')
>```

```
Output:
    Training for 15 epoch(s)...
    Epoch:    1/15    Loss: 5.400850810050964
    ..
    ..
    ..
    Epoch:   15/15    Loss: 3.294407793521881
    
    Model Trained and Saved
```

---
#### Checkpoint

After running the above training cell, our model will be saved by name, `trained_rnn`, and if you save your notebook progress, **you can pause here and come back to this code at another time**. You can resume your progress by running the next cell, which will load in our word:id dictionaries _and_ load in your saved model by name!


>```python
>
>import torch
>import helper
>import problem_unittests as tests
>
>_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
>trained_rnn = helper.load_model('./save/trained_rnn')
>```

## Generate TV Script

With the network trained and saved, you'll use it to generate a new, "fake" Seinfeld TV script in this section.

#### Generate Text

To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. You'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!


>```python
>
>import torch.nn.functional as F
>
>def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
>    """
>    Generate text using the neural network
>    :param decoder: The PyTorch Module that holds the trained neural network
>    :param prime_id: The word id to start the first prediction
>    :param int_to_vocab: Dict of word id keys to word values
>    :param token_dict: Dict of puncuation tokens keys to puncuation values
>    :param pad_value: The value used to pad a sequence
>    :param predict_len: The length of text to generate
>    :return: The generated text
>    """
>    rnn.eval()
>    
>    # create a sequence (batch_size=1) with the prime_id
>    current_seq = np.full((1, sequence_length), pad_value)
>    current_seq[-1][-1] = prime_id
>    predicted = [int_to_vocab[prime_id]]
>    
>    for _ in range(predict_len):
>        if train_on_gpu:
>            current_seq = torch.LongTensor(current_seq).cuda()
>        else:
>            current_seq = torch.LongTensor(current_seq)
>        
>        # initialize the hidden state
>        hidden = rnn.init_hidden(current_seq.size(0))
>        
>        # get the output of the rnn
>        output, _ = rnn(current_seq, hidden)
>        
>        # get the next word probabilities
>        p = F.softmax(output, dim=1).data
>        if(train_on_gpu):
>            p = p.cpu() # move to cpu
>         
>        # use top_k sampling to get the index of the next word
>        top_k = 5
>        p, top_i = p.topk(top_k)
>        top_i = top_i.numpy().squeeze()
>        
>        # select the likely next word index with some element of randomness
>        p = p.numpy().squeeze()
>        word_i = np.random.choice(top_i, p=p/p.sum())
>        
>        # retrieve that word from the dictionary
>        word = int_to_vocab[word_i]
>        predicted.append(word)     
>        
>        # the generated word becomes the next "current sequence" and the cycle can continue
>        current_seq = np.roll(current_seq, -1, 1)
>        current_seq[-1][-1] = word_i
>    
>    gen_sentences = ' '.join(predicted)
>    
>    # Replace punctuation tokens
>    for key, token in token_dict.items():
>        ending = ' ' if key in ['\n', '(', '"'] else ''
>        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
>    gen_sentences = gen_sentences.replace('\n ', '\n')
>    gen_sentences = gen_sentences.replace('( ', '(')
>    
>    # return all the sentences
>    return gen_sentences
>```

### Generate a New Script

It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- a. "jerry"
- b. "elaine"
- c. "george"
- d. "kramer"

We can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script.


>```python
># run the cell multiple times to get different results!
>gen_length = 400 # modify the length to your preference
>prime_word = 'jerry' # name for starting the script
>
>pad_word = helper.SPECIAL_WORDS['PADDING']
>generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], >gen_length)
>print(generated_script)
>```
    
```
Output:
    jerry:.""
    
    jerry:(to jerry) oh, yeah, yeah.
    
    jerry:(to himself) you see?
    
    elaine: oh, yeah...
    
    jerry: hey.
    
    kramer: well, i got to be a very interesting driver.(george nods)
    
    kramer:(pointing at the table) you know, i just had to go to the bathroom. i was wondering if you want to go with him, i don't know what you want, i can't.
    
    george:(laughs) oh, no... i just got a little more than the best way to be in the bathroom?
    
    jerry: no, it's all the way.
    
    george: oh, well. you don't have to go.
    
    jerry:(confused) what?
    
    elaine: well, you know, you know..
    
    jerry:(pointing at the counter) hey.
    
    jerry: hey!
    
    elaine:(quietly) yeah. i got it. i don't want to have any money.
    
    elaine:(pointing out) what is this?
    
    george: i can't. i don't know...
    
    george: oh, hi. hi jerry.
    
    jerry: hi, hi.
    
    elaine: hi, jerry.
    
    elaine:(shouting) what are you talking about?
    
    jerry: yeah, i don't know what i do.
    
    george: i don't know. i don't know...
    
    jerry: i don't have it. i mean, i don't want to be able to tell you this. you know what i want to say,"
    
    george:" what happened?
    
    kramer: well, i don't think you know, i just had a little.
    
    kramer: yeah, i think i got the tape.
    
    jerry: i don't know. but i don't know how to have a good time.
    
    elaine: oh my god. you want to know?
    
    jerry: well, i just don't want to have a
```

---
#### Save your favorite scripts

Once you have a script that you like (or find interesting), save it to a text file!


>```python
># save script to a text file
>f =  open("generated_script_1.txt","w")
>f.write(generated_script)
>f.close()
>```

### The TV Script is Not Perfect

It's ok if the TV script doesn't make perfect sense. It should look like alternating lines of dialogue, here is one such example of a few generated lines.

#### Example generated script

>jerry: what about me?
>
>jerry: i don't have to wait.
>
>kramer:(to the sales table)
>
>elaine:(to jerry) hey, look at this, i'm a good doctor.
>
>newman:(to elaine) you think i have no idea of this...
>
>elaine: oh, you better take the phone, and he was a little nervous.
>
>kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.
>
>jerry: oh, yeah. i don't even know, i know.
>
>jerry:(to the phone) oh, i know.
>
>kramer:(laughing) you know...(to jerry) you don't know.

---
We can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data.  

### Conclusion

We've successfully developed an algorithm that generates original TV scripts by training on Seinfeld dialogue data. The model was initially built using RNNs, and we leveraged pre-processing functions to prepare the data for training. We then implemented the RNN Module and forward and backpropagation functions to build the neural network. Finally, we trained the network using a train loop and hyperparameters.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Generate-TV-Scripts).




