package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFLiteObjectDetectionAPIModel implements ClassifierTwo {
    private static final String TAG = "TFLiteObjectDetectionAPIModelWithInterpreter";

    private static final int NUM_DETECTIONS = 10;
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    private static final int NUM_THREADS = 4;
    private boolean isModelQuantized;

    private int inputSize;

    private final List<String> labels = new ArrayList<>();
    private int[] intValues;

    private float[][][] outputLocations;
    private float[][] outputClasses;

    private float[][] outputScores;
    private float[] numDetections;

    private ByteBuffer imgData;

    private MappedByteBuffer tfLiteModel;
    private Interpreter.Options tfLiteOptions;
    private Interpreter tfLite;

    private float[][] foodMap;
    private static final int FOOD_SIZE = 4;
    private static final float WHITE_THRESH = 255f;

    private TFLiteObjectDetectionAPIModel() {}
    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public static ClassifierTwo create(
            final Context context,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException {
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

        AssetManager am = context.getAssets();
        InputStream is = am.open(labelFilename);

        MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
        MetadataExtractor metadata = new MetadataExtractor(modelFile);
        try (BufferedReader br =
                     new BufferedReader(
                             new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                //Log.w(TAG, line);
                d.labels.add(line);
            }
        }

        d.inputSize = inputSize;

        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(NUM_THREADS);
            options.setUseXNNPACK(true);
            d.tfLite = new Interpreter(modelFile, options);
            d.tfLiteModel = modelFile;
            d.tfLiteOptions = options;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];

        d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];
        return d;
    }

    @Override
    public String recognizeImage1(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage1");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];


                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model

                    // divide image pixels by 255 to get values in range from 0 to 1

//          imgData.putFloat(((pixelValue >> 16) & 0xFF) / WHITE_THRESH);
//          imgData.putFloat(((pixelValue >> 8) & 0xFF) / WHITE_THRESH);
//          imgData.putFloat((pixelValue & 0xFF)  / WHITE_THRESH);

                    //doing the color change here from RGB to BGR

                    imgData.putFloat((pixelValue & 0xFF) / WHITE_THRESH);
                    imgData.putFloat(((pixelValue >> 8) & 0xFF) / WHITE_THRESH);
                    imgData.putFloat(((pixelValue >> 16) & 0xFF) / WHITE_THRESH);
                }
            }
        }

        // convert RGB to BGR

        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
//    outputLocations = new float[1][NUM_DETECTIONS][4];
//    outputClasses = new float[1][NUM_DETECTIONS];
//    outputScores = new float[1][NUM_DETECTIONS];
//    numDetections = new float[1];


        Object[] inputArray = {imgData};

        Map<Integer, Object> outputMap = new HashMap<>();

        foodMap = new float[1][FOOD_SIZE];

//    outputMap.put(0, outputLocations);
//    outputMap.put(1, outputClasses);
//    outputMap.put(2, outputScores);
//    outputMap.put(3, numDetections);

        outputMap.put(0, foodMap);


        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");


        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


        Log.w("traytype", "!!!traytype: " + foodMap[0][0] + "|" + foodMap[0][1] +  "|" + foodMap[0][2]+ "|" + foodMap[0][3]);

        Integer foodInd = 0;
        Integer genderInd = 0;

        Float max = -1f;
        for (int i = 0; i < foodMap.length; i++) {
            float currValue = foodMap[0][i];

            if (currValue > max) {
                max = currValue;
                foodInd = i;
            }

        }





        //LOGGER.i("IND AGE: " + ageInd + " | IND Gender: " + genderInd);

   //     List<String> genderList = Arrays.asList("Female", "Male");
     //ac   List<String> foodList = Arrays.asList("Soup", "MainCourse", "Dessert", "Salad");

     //ac   String label = "food type: " + foodList.get(foodInd) ;
       // LOGGER.i("IND AGE: " + ageInd + " | IND Gender: " + genderInd);

     //   List<String> genderList = Arrays.asList("Female", "Male");
       // List<String> ageList = Arrays.asList("0-14yo", "15-40yo", "41-60yo", "61-100yo");
          List<String> foodList = Arrays.asList("soup", "mainCourse", "dessert", "salad");


        String label = "foodtype: " + foodList.get(foodInd);

        Trace.endSection();

        final int numDetectionsOutput = 1;
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        //Recognition rec = new Recognition("0", label, new RectF());

        //recognitions.add( rec );


        Trace.endSection();
        return label;
    }
//    @Override
//    public List<Recognition> recognizeImage(Bitmap bitmap) {
//        return null;
//    }



    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return null;
    }

    @Override
    public void close() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }
    }

    @Override
    public void setNumThreads(int num_threads) {
        if (tfLite != null) {
            tfLiteOptions.setNumThreads(num_threads);
            recreateInterpreter();
        }
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) {
            tfLiteOptions.setUseNNAPI(isChecked);
            recreateInterpreter();
        }
    }

    private void recreateInterpreter() {
        tfLite.close();
        tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
    }

    @Override
    public float getObjThresh() {
        return 0;
    }

}
