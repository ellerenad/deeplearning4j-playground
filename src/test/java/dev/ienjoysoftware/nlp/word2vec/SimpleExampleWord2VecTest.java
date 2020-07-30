package dev.ienjoysoftware.nlp.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;

import static org.junit.jupiter.api.Assertions.*;

class SimpleExampleWord2VecTest {

    SimpleExampleWord2Vec simpleExampleWord2Vec = new SimpleExampleWord2Vec();

    @Test
    void testTrainWord2Vec_export_import() throws FileNotFoundException {
       Word2Vec word2VecModel = simpleExampleWord2Vec.trainWord2Vec(
               SimpleExampleWord2Vec.WORD2VEC_INPUT_FILE_PATH,
               SimpleExampleWord2Vec.WORD2VEC_OUTPUT_FILE_PATH);
        assertNotNull(word2VecModel);

        Word2Vec word2VecModelLoaded = WordVectorSerializer.readWord2VecModel(new File( SimpleExampleWord2Vec.WORD2VEC_OUTPUT_FILE_PATH));
        assertNotNull(word2VecModelLoaded);
    }
}
