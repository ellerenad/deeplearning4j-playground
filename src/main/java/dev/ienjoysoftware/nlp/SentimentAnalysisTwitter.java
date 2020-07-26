package dev.ienjoysoftware.nlp;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.FileNotFoundException;
import java.util.logging.Logger;

public class SentimentAnalysisTwitter {
    private final static Logger log = Logger.getLogger(SentimentAnalysisTwitter.class.getName());

    // This file is taken from the examples downloaded by deeplearning4j library
    private final static String WORD2VEC_INPUT_FILE_PATH = "assets/nlp/word2vec/raw_sentences.txt";

    public Word2Vec trainWord2Vec() throws FileNotFoundException {
        // Inspired from org.deeplearning4j.examples.advanced.modelling.embeddingsfromcorpus.word2vec.Word2VecRawTextExample

        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(WORD2VEC_INPUT_FILE_PATH);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model");
        Word2Vec word2vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model");
        word2vec.fit();
        log.info("Word2vec model fitted");

        return word2vec;

        // Uncomment this to export the model to a file
        // log.info("Writing word vectors to text file....");
        // WordVectorSerializer.writeWord2VecModel(vec, "pathToSaveModel.txt");
    }

}
