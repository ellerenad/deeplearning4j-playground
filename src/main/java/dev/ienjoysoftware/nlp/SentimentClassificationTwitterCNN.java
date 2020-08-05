/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package dev.ienjoysoftware.nlp;

import dev.ienjoysoftware.nlp.util.DataSetIteratorWrapper;
import dev.ienjoysoftware.nlp.util.DataWrapper;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.Format;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882
 * <p>
 * Specifically, this is the 'static' model from there
 *
 * @author Alex Black
 */
public class SentimentClassificationTwitterCNN {

    // TODO refactor. Make configurable
    static String wordVectorsPath = "/Users/enriquedominguez/dl4j-examples-data/w2vec300/GoogleNews-vectors-negative300.bin.gz";
    // static String twitterDataSet = "/Users/enriquedominguez/dl4j-examples-data/sentiment_analysis_twitter/training.1600000.processed.noemoticon.csv";
    // static String twitterDataSet = "/Users/enriquedominguez/dl4j-examples-data/sentiment_analysis_twitter/training_reduced_800000.csv";
    static String twitterDataSet = "/Users/enriquedominguez/dl4j-examples-data/sentiment_analysis_twitter/training_reduced_2000.csv";

    public void runExample() throws Exception {

        // TODO Copy over to this class?
        // ImdbReviewClassificationRNN.checkDownloadW2VECModel();

        //Download and extract data
        // TODO Copy over to this class?
        // ImdbReviewClassificationRNN.downloadData();

        //Basic configuration
        int batchSize = 32;
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); //For shuffling repeatability

        //Set up the network configuration. Note that we have multiple convolution layers, each wih filter
        //widths of 3, 4 and 5 as per Kim (2014) paper.

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3, vectorSize)
                        .stride(1, vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4, vectorSize)
                        .stride(1, vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5, vectorSize)
                        .stride(1, vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                //MergeVertex performs depth concatenation on activations: 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                //Global pooling: pool over x/y locations (dimensions 2 and 3): Activations [minibatch,300,length,300] to [minibatch, 300]
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(2)    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                //Input has shape [minibatch, channels=1, length=1 to 256, 300]
                .setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Number of parameters by layer:");
        for (Layer l : net.getLayers()) {
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
        System.out.println("Creating DataSetIterators");
        DataSetIteratorWrapper dataSetIteratorWrapper = getDataSetIterator(wordVectors, batchSize, truncateReviewsToLength, rng);

        DataSetIterator trainIter = dataSetIteratorWrapper.getTrain();
        DataSetIterator testIter = dataSetIteratorWrapper.getTest();

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
        net.fit(trainIter, nEpochs);

        net.save(new File("/Users/enriquedominguez/development/mimacom/deeplearning4j-examples/dl4j-examples/models/sentimentClassificationTwitterCNN"));

        //After training: test a single sentence and generate a prediction
        String negativeSentence = "Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?";
        predictSingle(net, testIter, negativeSentence);

        String positiveSentence = "the sun is shining and i'm off for a driving lesson ";
        predictSingle(net, testIter, positiveSentence);
    }

    private static void predictSingle(ComputationGraph net, DataSetIterator testIter, String sentence) {
        INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator) testIter).loadSingleSentence(sentence);

        INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
        List<String> labels = testIter.getLabels();

        System.out.println("\n\nPredictions for sentence: " + sentence);
        for (int i = 0; i < labels.size(); i++) {
            System.out.println("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
        }
    }


    @SuppressWarnings("Duplicates")
    DataSetIteratorWrapper getDataSetIterator(WordVectors wordVectors, int minibatchSize,
                                                      int maxSentenceLength, Random rng) throws IOException {
        // TODO Receive as param
        String path = twitterDataSet;

        DataWrapper dataWrapper = getCleanData(path);

        // The sentence providers shuffle the data internally
        LabeledSentenceProvider trainingSentenceProvider = new CollectionLabeledSentenceProvider(dataWrapper.getTraining_data(), dataWrapper.getTraining_labels(), rng);
        LabeledSentenceProvider testSentenceProvider = new CollectionLabeledSentenceProvider(dataWrapper.getTest_data(), dataWrapper.getTest_labels(), rng);


        DataSetIterator trainingDataSetIterator = new CnnSentenceDataSetIterator.Builder(Format.CNN2D)
                .sentenceProvider(trainingSentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();

        DataSetIterator testDataSetIterator = new CnnSentenceDataSetIterator.Builder(Format.CNN2D)
                .sentenceProvider(testSentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(minibatchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .build();

        return new DataSetIteratorWrapper(trainingDataSetIterator, testDataSetIterator);
    }



    DataWrapper getCleanData(String path) throws IOException {
        List<String> trainingLabelList = new ArrayList<>();
        List<String> trainingDataList = new ArrayList<>();
        List<String> testLabelList = new ArrayList<>();
        List<String> testDataList = new ArrayList<>();
        // Load dataset
        LineIterator lineIterator = FileUtils.lineIterator(new File(path), "UTF-8");
        int counter = 0;
        String line = null;
        while (lineIterator.hasNext()) {
            line = lineIterator.nextLine();
            try {
                String[] parts = line.split("\",\"");
                String label = getLabel(parts);
                String data = getData(parts);
                // TODO Check splitting
                if (counter % 2 == 0) {
                    trainingLabelList.add(label);
                    trainingDataList.add(data);
                } else {
                    testLabelList.add(label);
                    testDataList.add(data);
                }
            } catch(Exception ex){
                System.err.println("Exception: " + ex + ". Counter = "+ counter +". Line = "+line);
            }
            counter++;

            // TODO Delete? limit the amount of data to load - this is a small test
            if(counter % 1000 == 0){
                System.out.println("Read line number "+ counter + " from dataset. Line: "+ line);
            }

        }
        System.out.println("End: Read line number "+ counter + " from dataset.Line: "+ line);
        System.out.println("trainingLabelList\ttrainingDataList\ttestLabelList\ttestDataList\n" +
                trainingLabelList.size() + "\t"+ trainingDataList.size() + "\t"+ testLabelList.size() + "\t"+ testDataList.size());


        return new DataWrapper(trainingLabelList, trainingDataList, testLabelList, testDataList);
    }

    String getLabel(String[] parts){
        // TODO make null check, make length check
        return parts[0].replace("\"", "").equals("0") ? "Negative" : "Positive";
    }

    String getData(String[] parts){
        String data = null;

            if (parts.length == 6) {
                data = parts[5];
            } else if (parts.length >= 6) {
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 5; i < parts.length; i++) {
                    stringBuilder.append(parts[i]);
                }
                data = stringBuilder.toString();
            }
        // TODO Check: it could be that i am replacing too much of "
        return data.replace("\"", "");
    }


}
