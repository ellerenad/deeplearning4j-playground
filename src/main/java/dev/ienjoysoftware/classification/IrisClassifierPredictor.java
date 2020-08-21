package dev.ienjoysoftware.classification;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.buffer.FloatBuffer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class IrisClassifierPredictor {
    private static Logger log = LoggerFactory.getLogger(IrisClassifierPredictor.class);

    private final static int FIELDS_COUNT = 4;
    private static final int CLASSES_COUNT = 3;

    private static int INDEX_SEPAL_LENGTH = 0;
    private static int INDEX_SEPAL_WIDTH = 1;
    private static int INDEX_PETAL_LENGTH = 2;
    private static int INDEX_PETAL_WIDTH = 3;
    static final List<String> LABELS = Arrays.asList("Iris Setosa", "Iris Versicolour", "Iris Virginica");

    private static final String STORED_NORMALIZER_FILENAME = "normalizer";
    private static final String STORED_MODEL_FILENAME = "trainedModel.zip";

    private MultiLayerNetwork model;
    private DataNormalization dataNormalizer;

    /**
     * The constructor loads the required artifacts (model and normalizer) to perform the prediction.
     * @param savedModelBasePath
     * @throws Exception
     */
    public IrisClassifierPredictor(String savedModelBasePath) throws Exception {
        model = loadModel(savedModelBasePath);
        dataNormalizer = loadNormalizer(savedModelBasePath);
    }

    /**
     * Predict a label given a domain object
     *
     * @param iris representation of the iris flower
     * @return the predicted label
     */
    public String classify(Iris iris) {

        // Transform the data to the required format
        INDArray indArray = getArray(iris);

        // Normalize the data the same way it was normalized in the training phase
        dataNormalizer.transform(indArray);

        // Do the prediction
        INDArray result = model.output(indArray, false);

        // Get the index with the greatest probabilities
        int predictedLabelIndex = getIndexPredictedLabel(result);
        log.debug("predictedLabelIndex= {}", predictedLabelIndex);
        return LABELS.get(predictedLabelIndex);
    }


    private MultiLayerNetwork loadModel(String basePath) throws IOException {
        File locationToSaveModel = new File(basePath + STORED_MODEL_FILENAME);
        MultiLayerNetwork restoredModel = MultiLayerNetwork.load(locationToSaveModel, false);
        return restoredModel;
    }

    private DataNormalization loadNormalizer(String basePath) throws Exception {
        File locationToSaveNormalizer = new File(basePath + STORED_NORMALIZER_FILENAME);
        DataNormalization restoredNormalizer = NormalizerSerializer.getDefault().restore(locationToSaveNormalizer);
        return restoredNormalizer;
    }

    /**
     * Transform the data from the domain to the object required by the library
     *
     * @param iris representation of the iris flower
     * @return an INDArray the framework can work with
     */
    private static INDArray getArray(Iris iris) {
        // It is important to use float. Using double, the model would not work properly
        float[] input = new float[FIELDS_COUNT];
        input[INDEX_SEPAL_LENGTH] = iris.getSepal_length();
        input[INDEX_SEPAL_WIDTH] = iris.getSepal_width();
        input[INDEX_PETAL_LENGTH] = iris.getPetal_length();
        input[INDEX_PETAL_WIDTH] = iris.getPetal_width();

        NDArray ndArray = new NDArray(1, FIELDS_COUNT); // The empty constructor causes a NPE in add method
        DataBuffer dataBuffer = new FloatBuffer(input);
        ndArray.setData(dataBuffer);
        return ndArray;
    }


    /**
     * Get the index of the predicted label
     *
     * @param predictions INDArray with the probabilities per label
     * @return the index with the greatest probabilities
     */
    private static int getIndexPredictedLabel(INDArray predictions) {
        int maxIndex = 0;
        log.debug("Predictions = {}", predictions.toString(1, false, 5));
        // We should get max CLASSES_COUNT amount of predictions with probabilities.
        for (int i = 0; i < CLASSES_COUNT; i++) {
            if (predictions.getFloat(i) > predictions.getFloat(maxIndex)) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
