package dev.ienjoysoftware.image.classification;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.BaseLabels;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class Yolo2ImageClassifier {
    // Width required by the YOLO2 Model
    private static final int YOLO2_WIDTH = 416;

    // Height required by the YOLO2 Model
    private static final int YOLO2_HEIGHT = 416;

    // Minimum confidence required to accept a prediction
    private static final double DETECTION_THRESHOLD = .5;

    // non-maximum suppression which removes redundant overlapping bounding boxes
    private static final double NMS_THRESHOLD = .4;

    private static final int GRID_WIDTH = 13;
    private static final int GRID_HEIGHT = 13;

    // Labels supported by the YOLO2 model
    // All the labels are at https://github.com/pjreddie/darknet/blob/master/data/coco.names
    private static BaseLabels labels;

    private ZooModel yolo2Model;
    private ComputationGraph pretrainedComputationGraph;

    // To load the image with the YOLO requirements: width, height, color scheme
    private NativeImageLoader yoloImageLoader;

    // To load the original image
    private NativeImageLoader imageLoader;

    public Yolo2ImageClassifier() throws IOException {
        yolo2Model = YOLO2.builder().build();
        pretrainedComputationGraph = (ComputationGraph) yolo2Model.initPretrained();
        imageLoader = new NativeImageLoader();
        yoloImageLoader = new NativeImageLoader(YOLO2_WIDTH, YOLO2_HEIGHT, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        labels = new COCOLabels();
    }

    public List<DetectedObject> classify(String inputImagePath, String outputImagePath) throws IOException {

        // Load the image from disk
        File fileOriginalImage = new File(inputImagePath);
        INDArray iNDArrayOriginalImage = imageLoader.asMatrix(fileOriginalImage);

        // Resize the image to match the required size by YOLO2
        Mat matResizedImage = yoloImageLoader.asMat(iNDArrayOriginalImage);

        // Scale the images, as in "normalize the pixels to be on the range from 0 to 1"
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        INDArray iNDArrayTransformedImage = yoloImageLoader.asMatrix(matResizedImage);
        scaler.transform(iNDArrayTransformedImage);

        // Perform the classification
        INDArray outputs = pretrainedComputationGraph.outputSingle(iNDArrayTransformedImage);
        List<DetectedObject> detectedObjects = YoloUtils.getPredictedObjects
                (Nd4j.create(((YOLO2) yolo2Model).getPriorBoxes()),
                        outputs,
                        DETECTION_THRESHOLD,
                        NMS_THRESHOLD);

        // Annotate the original image
        Image originalImage = imageLoader.asImageMatrix(fileOriginalImage);
        int originalWidth = originalImage.getOrigW();
        int originalHeight = originalImage.getOrigH();
        annotate(originalWidth, originalHeight, matResizedImage, detectedObjects, outputImagePath);

        return detectedObjects;
    }

    private void annotate(int imageWidth, int imageHeight, Mat rawImage, List<DetectedObject> detectedObjects, String outputImagePath) {
        for (DetectedObject detectedObject : detectedObjects) {
            // Mark the detected objects with a rectangle and its label
            // Calculate the positions of the corners of the rectangle
            double[] xy1 = detectedObject.getTopLeftXY();
            double[] xy2 = detectedObject.getBottomRightXY();
            String label = labels.getLabel(detectedObject.getPredictedClass());
            int x1 = (int) Math.round(imageWidth * xy1[0] / GRID_WIDTH);
            int y1 = (int) Math.round(imageHeight * xy1[1] / GRID_HEIGHT);
            int x2 = (int) Math.round(imageWidth * xy2[0] / GRID_WIDTH);
            int y2 = (int) Math.round(imageHeight * xy2[1] / GRID_HEIGHT);
            // Draw the rectangle
            rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
            // Draw the label
            putText(rawImage, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.RED);
            // Store the file on disk
            imwrite(outputImagePath, rawImage);
        }
    }

}
