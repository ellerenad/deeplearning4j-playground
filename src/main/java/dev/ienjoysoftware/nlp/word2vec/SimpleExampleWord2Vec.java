package dev.ienjoysoftware.nlp.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.FileNotFoundException;
import java.util.logging.Logger;

public class SimpleExampleWord2Vec {
    private final static Logger log = Logger.getLogger(SimpleExampleWord2Vec.class.getName());

    // This file is inspired from the examples downloaded by deeplearning4j library
    // Download the following and configure the path belo http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    public final static String WORD2VEC_INPUT_FILE_PATH = "assets/nlp/word2vec/raw_sentences.txt";
    public final static String WORD2VEC_OUTPUT_FILE_PATH = "models/word2vec/SimpleExampleWord2Vec";

    /**
     * Dummy / convenience method calling the underlying trainWord2Vec with predefined constant values
     * @return
     * @throws FileNotFoundException
     */
    public Word2Vec trainWord2Vec() throws FileNotFoundException {
        return this.trainWord2Vec(WORD2VEC_INPUT_FILE_PATH, WORD2VEC_OUTPUT_FILE_PATH);
    }

    /**
     * Train a {@link org.deeplearning4j.models.word2vec.Word2Vec} model, based on the given vocabulary.
     * The model is also exported.
     * @param input_path The path of the vocabulary to train this model
     * @param output_path The path where the trained model will be exported to
     * @return the trained {@link org.deeplearning4j.models.word2vec.Word2Vec} model
     * @throws FileNotFoundException the input file was not found
     */
    public Word2Vec trainWord2Vec(String input_path, String output_path) throws FileNotFoundException {
        // Inspired from org.deeplearning4j.examples.advanced.modelling.embeddingsfromcorpus.word2vec.Word2VecRawTextExample

        SentenceIterator sentenceIterator = new BasicLineIterator(input_path);
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor strips off all numbers, punctuation symbols and some special symbols.
            It forces lower case for all tokens.
         */
        tokenizer.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building Word2Vec model");
        Word2Vec word2vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizer)
                .build();

        log.info("Fitting Word2Vec model");
        word2vec.fit();
        log.info("Word2vec model fitted");

        log.info("Writing word vectors to a file. [path={ "+output_path+"}]");
        WordVectorSerializer.writeWord2VecModel(word2vec, output_path);

        return word2vec;
    }

}
