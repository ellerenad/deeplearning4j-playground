test 1:

File: training_reduced_2000.csv
========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.6758
 Precision:       0.7479
 Recall:          0.5283
 F1 Score:        0.6192
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
   0   1
---------
 408  88 | 0 = 0
 233 261 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


Predictions for sentence: Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?
P(0) = 0.7091114521026611
P(4) = 0.2908885180950165


Predictions for sentence: the sun is shining and i'm off for a driving lesson
P(0) = 0.5061236023902893
P(4) = 0.4938764274120331

execution time (milliseconds): 173817


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


File: training_reduced_2000.csv
========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9948
 Precision:       0.0000
 Recall:          0.0000
 F1 Score:        0.0000
Precision, recall & F1: reported for positive class (class 1 - "1") only

Warning: 1 class was never predicted by the model and was excluded from average precision
Classes excluded from average precision: [1]

=========================Confusion Matrix=========================
    0    1
-----------
 4969    0 | 0 = 0
   26    0 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


Predictions for sentence: Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?
P(0) = 0.9987585544586182
P(4) = 0.0012414384400472045


Predictions for sentence: the sun is shining and i'm off for a driving lesson
P(0) = 0.9967586398124695
P(4) = 0.003241364611312747
Aug 01, 2020 11:41:40 PM dev.ienjoysoftware.nlp.SentimentClassificationTwitterCNNTest testTrainNetwork
INFO: execution time (milliseconds): 182812



+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

training_reduced_40000


========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9914
 Precision:       0.0000
 Recall:          0.0000
 F1 Score:        0.0000
Precision, recall & F1: reported for positive class (class 1 - "1") only

Warning: 1 class was never predicted by the model and was excluded from average precision
Classes excluded from average precision: [1]

=========================Confusion Matrix=========================
    0    1
-----------
 9933    0 | 0 = 0
   86    0 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


Predictions for sentence: Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?
P(0) = 0.9999810457229614
P(4) = 1.895870991575066E-5


Predictions for sentence: the sun is shining and i'm off for a driving lesson
P(0) = 0.9993531107902527
P(4) = 6.469375803135335E-4
Aug 02, 2020 12:02:15 AM dev.ienjoysoftware.nlp.SentimentClassificationTwitterCNNTest testTrainNetwork
INFO: execution time (milliseconds): 180146



+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting here, the bug was fixed
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/Users/enriquedominguez/dl4j-examples-data/sentiment_analysis_twitter/training_reduced_800000.csv

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.7472
 Precision:       0.7200
 Recall:          0.8065
 F1 Score:        0.7608
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
     0     1
-------------
 68358 30966 | 0 = 0
 19106 79630 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


Predictions for sentence: Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?
P(Negative) = 0.9179206490516663
P(Positive) = 0.08207931369543076


Predictions for sentence: the sun is shining and i'm off for a driving lesson
P(Negative) = 0.48016732931137085
P(Positive) = 0.5198327302932739
Aug 05, 2020 10:31:58 PM dev.ienjoysoftware.nlp.SentimentClassificationTwitterCNNTest testTrainNetwork
INFO: execution time (milliseconds): 1581209

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/Users/enriquedominguez/dl4j-examples-data/sentiment_analysis_twitter/training_reduced_2000.csv

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.6721
 Precision:       0.6382
 Recall:          0.7886
 F1 Score:        0.7055
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
   0   1
---------
 138 110 | 0 = 0
  52 194 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


Predictions for sentence: Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?
P(Negative) = 0.900755763053894
P(Positive) = 0.09924420714378357


Predictions for sentence: the sun is shining and i'm off for a driving lesson
P(Negative) = 0.421742707490921
P(Positive) = 0.5782573223114014
Aug 05, 2020 11:00:33 PM dev.ienjoysoftware.nlp.SentimentClassificationTwitterCNNTest testTrainNetwork
INFO: execution time (milliseconds): 142727

Process finished with exit code 0







