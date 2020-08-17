package dev.ienjoysoftware.image.classification;

import org.apache.commons.lang.time.StopWatch;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertFalse;

class Yolo2ImageClassifierTest {
    private final static Logger log = Logger.getLogger(Yolo2ImageClassifierTest.class.getName());

    @Test
    void classify() throws IOException {
        Yolo2ImageClassifier yolo2ImageClassifier = new Yolo2ImageClassifier();
        List<String> file_paths = Arrays.asList(
                "1 - harley-davidson-xAHtaYIHlPI-unsplash.jpg",
                "2 - anthony-renovato-6HxC-fZjlI0-unsplash.jpg",
                "3 - mikhail-vasilyev-gGC63oug3iY-unsplash.jpg",
                "4 - Polar_Bear.jpg",
                "5 - rahul-dey-kzQ6gbTR-Fg-unsplash.jpg",
                "6 - andre-hunter-p-I9wV811qk-unsplash.jpg",
                "7 - Brown_bear.jpg",
                "8 - Bear.jpg"
        );

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (String filePath : file_paths) {
            log.info("processing: "+ filePath);
            String inputPath = "assets/images/input/" + filePath;
            String outputPath = "assets/images/output/"+ filePath.replace(".jpg", "-annotated.jpg");

            List<DetectedObject> detectedObjects = yolo2ImageClassifier.classify(inputPath, outputPath);
            assertFalse(detectedObjects.isEmpty());
        }
        stopWatch.stop();
        log.info("Execution time (milliseconds): "+  stopWatch.getTime());
    }
}
