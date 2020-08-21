package dev.ienjoysoftware.classification;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IrisClassifierPredictorTest {

    /**
     * This test assumes the existence of a model on a given path.
     * The model can be generated using {@link IrisClassifierTrainerTest#testTrain()} .
     * It could be said this is a bad practice, because it assumes an order on the tests, which cannot be assured
     * but I left like this it for easier demo purposes.
     */
    @Test
    void classify() throws Exception {
        String modelPath =  "models/iris_classification/unit_test/";
        IrisClassifierPredictor irisClassifierPredictor = new IrisClassifierPredictor(modelPath);

        Iris iris0 = new Iris(4.5f,2.3f,1.3f,0.3f);
        String label0 = irisClassifierPredictor.classify(iris0);
        assertEquals(label0, "Iris Setosa");

        Iris iris1 = new Iris(6.2f,2.2f,4.5f,1.5f);
        String label1 = irisClassifierPredictor.classify(iris1);
        assertEquals(label1, "Iris Versicolour");


        Iris iris2 = new Iris(6.2f,3.4f,5.4f,2.3f);
        String label2 = irisClassifierPredictor.classify(iris2);
        assertEquals(label2, "Iris Virginica");
    }
}
