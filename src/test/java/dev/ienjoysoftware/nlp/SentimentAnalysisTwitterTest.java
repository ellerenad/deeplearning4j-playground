package dev.ienjoysoftware.nlp;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.jupiter.api.Test;

import java.io.FileNotFoundException;

import static org.junit.jupiter.api.Assertions.*;

class SentimentAnalysisTwitterTest {

    SentimentAnalysisTwitter sentimentAnalysisTwitter = new SentimentAnalysisTwitter();

    @Test
    public void testTrainWord2Vec() throws FileNotFoundException {
        Word2Vec word2Vec = sentimentAnalysisTwitter.trainWord2Vec();
        assertNotNull(word2Vec);
    }



}