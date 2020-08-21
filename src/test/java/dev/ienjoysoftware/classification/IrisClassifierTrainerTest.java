package dev.ienjoysoftware.classification;

import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertTrue;

class IrisClassifierTrainerTest {
    private static Logger log = LoggerFactory.getLogger(IrisClassifierTrainer.class);
    private static double MIN_ACCEPTABLE_ACCURACY = 0.90; // Arbitrary number ;)



    @Test
    void testTrain() throws Exception {

        String irisDataset = "iris.txt";
        String irisDatasetPath = getClass().getClassLoader().getResource(irisDataset).getPath();

        String outputPath = "models/iris_classification/";
        IrisClassifierTrainer irisClassifierTrainer = new IrisClassifierTrainer(outputPath);

        Evaluation evaluation = irisClassifierTrainer.train(irisDatasetPath, "unit_test");

        assertTrue(evaluation.accuracy() > MIN_ACCEPTABLE_ACCURACY);

    }
}
