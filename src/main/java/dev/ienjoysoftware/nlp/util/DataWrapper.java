package dev.ienjoysoftware.nlp.util;

import java.util.List;

public class DataWrapper {
    List<String> training_labels;
    List<String> training_data;
    List<String> test_labels;
    List<String> test_data;

    public DataWrapper(List<String> training_labels, List<String> training_data, List<String> test_labels, List<String> test_data) {
        this.training_labels = training_labels;
        this.training_data = training_data;
        this.test_labels = test_labels;
        this.test_data = test_data;
    }

    public List<String> getTraining_labels() {
        return training_labels;
    }

    public List<String> getTraining_data() {
        return training_data;
    }

    public List<String> getTest_labels() {
        return test_labels;
    }

    public List<String> getTest_data() {
        return test_data;
    }
}
