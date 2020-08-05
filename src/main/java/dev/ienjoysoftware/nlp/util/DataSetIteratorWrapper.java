package dev.ienjoysoftware.nlp.util;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DataSetIteratorWrapper {

    private DataSetIterator train;
    private DataSetIterator test;

    public DataSetIteratorWrapper(DataSetIterator train, DataSetIterator test) {
        this.train = train;
        this.test = test;
    }

    public DataSetIterator getTrain() {
        return train;
    }

    public DataSetIterator getTest() {
        return test;
    }
}
