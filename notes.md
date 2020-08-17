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

- memory issues:
 params for the VM
 -Xms1024m
 -Xmx10g
 -XX:MaxPermSize=2g

reduce dataset:
head -1000 training.1600000.processed.noemoticon.csv >> training_reduced.csv
tail -1000 training.1600000.processed.noemoticon.csv >> training_reduced.csv

count lines in a file:
wc -l

Difficulties:
- The java.util.Scanner would read just up to the half of the file (seen on files bigger than 40,000 lines), messing with the training. Don't know why

Plan:

- show a way to use the example
- extend the example to twitter + more labels


downloaded data set https://www.kaggle.com/kazanova/sentiment140
despite the description of kaggel, it has just 2 labels: 0 (negative) and 4 (positive)
extract some examples for easy experimentation with

grep -m1 \"0\" training*.csv >> file
grep -m1 \"4\" training*.csv >> file

- supports early stopping with EarlyStoppingConfiguration (it is a wrapper for the training)

- Examples are hard to find
    - Documentation and explanation could be better
    - Some examples were moved and the google index ends up in not found code
- good or bad? : some examples do it the hard way and other the easy way, but it is hard to find out whether this is the easy way or not      


What are the differences between a model and a computation graph?


images from unsplash:
motorcycles: <span>Photo by <a href="https://unsplash.com/@harleydavidson?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Harley-Davidson</a> on <a href="https://unsplash.com/s/photos/free?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
Parrot <span>Photo by <a href="https://unsplash.com/@miklevasilyev?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">MIKHAIL VASILYEV</a> on <a href="https://unsplash.com/s/photos/free?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
<span>Photo by <a href="https://unsplash.com/@dre0316?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Andre Hunter</a> on <a href="https://unsplash.com/s/photos/free?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
 <span>Photo by <a href="https://unsplash.com/@dynamo10?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Rahul Dey</a> on <a href="https://unsplash.com/s/photos/free?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
Bear <span>Photo by <a href="https://unsplash.com/@trill6124?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">anthony renovato</a> on <a href="https://unsplash.com/s/photos/bear?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
