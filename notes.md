# Personal notes for the talk at the Java Forum Stuttgart 2020

- enables transfer learning by providing a zoo model -> https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning
- import from keras is possible?
- sentiment analyisis: word2vec, LSTM
- look for the UiServer tool
- allows importing models trained with keras and with tf



-----------------------------
Sentiment analysis
1.- define and train a word2vec model
2.- define a neural network
3.- use the iterator provided on the word2vec model to train the neural network - not exactly. It is more like on the data iterator, you need to give the word2vec model
4.- be happy :D

links:
https://deeplearning4j.konduit.ai/language-processing/word2vec
https://data.world/jaredfern/googlenews-reduced-200-d

FILE=gnews_mod.csv
tail -n +2 "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"

echo '1 2 3 4' | cat - "$FILE" > temp && mv temp "$FILE"

examples:
Word2VecRawTextExample.java
ImdbReviewClassificationRNN.java

Plan:

- show a way to use the example
- extend the example to twitter + more labels


downloaded data set https://www.kaggle.com/kazanova/sentiment140
despite the description of kaggel, it has just 2 labels: 0 (negative) and 4 (positive)
extract some examples for easy experimentation with

grep -m1 \"0\" training*.csv >> file
grep -m1 \"4\" training*.csv >> file
