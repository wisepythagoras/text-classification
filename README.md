# Text Classification With Keras

This repository contains code showing different methods of classifying text with the use of Tensorflow and Keras.

## Data

I've taken the data from [this dataset](https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment) and consolidated the text and labels in a single TSV file. Note that the text needs to be cleaned of all special characters (@#!$% etc). After that's done I placed the file in the `data` folder with the name `out.tsv`.

## Dense

This method uses a series of simple [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) and [PReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU?hl=en) layers with dropouts in between. Specifically, it starts with

``` python
out = Dense(512, activation='relu', input_shape=(max_words,))(inputs)
out = PReLU()(out)
out = Dropout(0.25)(out)
```

Then I've added three sets of hidden layers with decreasing units and dropout amount and ends with a dense layer which returns the expected output shape. 
