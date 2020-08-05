package dev.ienjoysoftware.nlp;

import dev.ienjoysoftware.nlp.util.DataWrapper;
import org.apache.commons.lang.time.StopWatch;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

class SentimentClassificationTwitterCNNTest {
    private final static Logger log = Logger.getLogger(SentimentClassificationTwitterCNNTest.class.getName());


    SentimentClassificationTwitterCNN sentimentClassificationTwitterCNN = new SentimentClassificationTwitterCNN();

    @Test
    public void testGetCleanData() throws IOException {
        String test_csv_path = "SentimentClassificationTwitterCNN_test_data_csv_read.csv";

        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(test_csv_path).getFile());
        String absolutePath = file.getAbsolutePath();


        DataWrapper dataWrapper = sentimentClassificationTwitterCNN.getCleanData(absolutePath);

        assertEquals(3, dataWrapper.getTraining_data().size());
        assertEquals(3, dataWrapper.getTraining_labels().size());
        assertEquals(3, dataWrapper.getTest_data().size());
        assertEquals(3, dataWrapper.getTest_labels().size());
        // Assert training dataset
        assertEquals("Negative",dataWrapper.getTraining_labels().get(0));
        assertEquals("Negative",dataWrapper.getTraining_labels().get(1));
        assertEquals("Positive",dataWrapper.getTraining_labels().get(2));

        assertEquals("about to file taxes ", dataWrapper.getTraining_data().get(0));
        assertEquals("@FakerPattyPattz Oh dear. Were you drinking out of the forgotten table drinks? ", dataWrapper.getTraining_data().get(1));
        assertEquals("one of my friend called me, and asked to meet with her at Mid Valley today...but i've no time *sigh* ", dataWrapper.getTraining_data().get(2));

        // Assert test dataset
        assertEquals("Negative",dataWrapper.getTest_labels().get(0));
        assertEquals("Positive",dataWrapper.getTest_labels().get(1));
        assertEquals("Positive",dataWrapper.getTest_labels().get(2));

        assertEquals("@LettyA ahh ive always wanted to see rent  love the soundtrack!!", dataWrapper.getTest_data().get(0));
        assertEquals("@alydesigns i was out most of the day so didn't get much done ", dataWrapper.getTest_data().get(1));
        assertEquals("I hate when I have to call and wake people up ", dataWrapper.getTest_data().get(2));

    }


    @Test
    public void testTrainNetwork() throws Exception {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        try {
            sentimentClassificationTwitterCNN.runExample();
        } finally {
            stopWatch.stop();
            log.info("execution time (milliseconds): " + stopWatch.getTime());
        }

    }


}
