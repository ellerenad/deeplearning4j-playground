package dev.ienjoysoftware.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

// Code inspired on org.deeplearning4j.examples.quickstart.modeling.feedforward.classification.IrisClassifier
public class IrisClassifierTrainer {
    private static Logger log = LoggerFactory.getLogger(IrisClassifierTrainer.class);

    private static final int CLASSES_COUNT = 3;
    private static final int LABEL_INDEX = 4;
    private static final int FEATURES_COUNT = 4;
    private static long SEED = 6;
    public static final int TRAIN_ITERATIONS = 1000;

    private static final int TOTAL_LINES = 150;
    private static final double TRAIN_TO_TEST_RATIO = 0.65;

    private static final String STORED_MODEL_FILENAME = "/trainedModel.zip";
    private static final String STORED_NORMALIZER_FILENAME = "/normalizer";

    private String modelOutputBasePath;

    public IrisClassifierTrainer(String modelOutputBasePath) {
        this.modelOutputBasePath = modelOutputBasePath;
    }

    /**
     * Perform the whole training process, consisting in:
     * - Load the data
     * - Prepare it: shuffle, normalize
     * - Split into test and training sets
     * - Configure and train the Neural Network
     * - Store the model and the normalizer
     *
     * @return an object to evaluate the performance of the training of the Neural Network
     * @throws IOException, InterruptedException
     */
    public Evaluation train(String datasetPath, String datasetIdentifier) throws IOException, InterruptedException {

        log.info("Loading dataset from {}", datasetPath);
        // Load data
        DataSet allData = loadData(datasetPath);

        // Shuffle the data. Important! otherwise, the model won't perform even remotely well
        allData.shuffle(SEED);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData); // Get stats about the data
        normalizer.transform(allData); // Transform the data by applying the normalization

        // Split in train and test datasets
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(TRAIN_TO_TEST_RATIO);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // Get configuration of the Neural Network
        MultiLayerConfiguration configuration = getMultiLayerConfiguration();

        // Train Neural Network
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(100)); //Print score every 100 parameter updates

        // Do TRAIN_ITERATIONS = 1000 iterations to train the model
        for(int x = 0; x < TRAIN_ITERATIONS; x++) {
            model.fit(trainingData);
        }

        // Save the model and the normalizer
        String modelOutputPath = modelOutputBasePath + datasetIdentifier + "/";
        store(model, normalizer, modelOutputPath);

        // Evaluate Neural Network
        Evaluation evaluation = evaluate(model, testData);
        log.info(evaluation.stats());
        return evaluation;
    }



    /**
     * Persist the required objects for later loading and prediction
     * @param model the trained model to be saved
     * @param normalizer the normalizer used with the data
     * @param outputPath the path where the artifacts will be saved
     * @throws IOException
     */
    private void store(MultiLayerNetwork model, DataNormalization normalizer, String outputPath) throws IOException {
        // Creating the folder to store the data
        File baseLocationToSaveModel = new File(outputPath);
        baseLocationToSaveModel.mkdirs();

        // Storing the model
        File locationToSaveModel = new File(outputPath + STORED_MODEL_FILENAME);
        model.save(locationToSaveModel, false);

        // Storing the normalizer
        File locationToSaveNormalizer = new File(outputPath + STORED_NORMALIZER_FILENAME);
        NormalizerSerializer.getDefault().write(normalizer, locationToSaveNormalizer);
        log.info("Model and normalizer stored at {}", outputPath);
    }

    /**
     * Evaluate the trained MultiLayerNetwork
     *
     * @param testData the previously separated data to perform the test on
     * @param model    the model to test
     * @return Evaluation object
     */
    private static Evaluation evaluate(MultiLayerNetwork model, DataSet testData) {
        INDArray output = model.output(testData.getFeatures());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        return eval;
    }

    /**
     * Load the data of a training set on a file
     *
     * @param path path relative to the classpath
     * @return a DataSet containing the data of the file
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet loadData(String path) throws IOException, InterruptedException {
        DataSet allData;
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new File(path)));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, TOTAL_LINES, LABEL_INDEX, CLASSES_COUNT);
            allData = iterator.next();
        }
        return allData;
    }

    /**
     * Get the configuration of the Neural Network
     */
    private static MultiLayerConfiguration getMultiLayerConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3) // The input layer must have FEATURES_COUNT = 4 nodes
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build()) // The output layer must have CLASSES_COUNT = 3 nodes
                .build();
    }

}
