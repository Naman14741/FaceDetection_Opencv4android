package app.cameraapp;

import android.util.Log;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.IOException;
import java.io.InputStream;

public class Loader {
    private final MainActivity mainActivity;
    private static final String TAG = "Loader";

    public Loader(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
    }

    public boolean loadOpenCV() {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
            return true;
        } else {
            Log.e(TAG, "OpenCV loaded failed!");
            Toast.makeText(mainActivity, "OpenCV was not found!", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    public boolean loadYuNetModel() {
        try (InputStream is = mainActivity.getResources().openRawResource(R.raw.face_detection_yunet_2023mar)) {
            int size = is.available();
            byte[] buffer = new byte[size];
            //noinspection ResultOfMethodCallIgnored
            is.read(buffer);
            MatOfByte mModelBuffer = new MatOfByte(buffer);
            MatOfByte mConfigBuffer = new MatOfByte();
            mainActivity.faceDetector = FaceDetectorYN.create("onnx", mModelBuffer, mConfigBuffer, new Size(320, 320));
            mainActivity.faceDetector.setScoreThreshold(0.8f);
            Log.i(TAG, "YuNet initialized successfully!");
            return true;
        } catch (IOException e) {
            Log.e(TAG, "YuNet loaded failed" + e);
            Toast.makeText(mainActivity, "YuNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    public boolean loadMobileFaceNetModel() {
        try (InputStream protoIs = mainActivity.getResources().openRawResource(R.raw.mobilefacenet);
             InputStream modelIs = mainActivity.getResources().openRawResource(R.raw.mobilefacenet_caffemodel)) {
            int protoSize = protoIs.available();
            byte[] protoBuffer = new byte[protoSize];
            //noinspection ResultOfMethodCallIgnored
            protoIs.read(protoBuffer);

            int modelSize = modelIs.available();
            byte[] modelBuffer = new byte[modelSize];
            //noinspection ResultOfMethodCallIgnored
            modelIs.read(modelBuffer);

            MatOfByte bufferProto = new MatOfByte(protoBuffer);
            MatOfByte bufferModel = new MatOfByte(modelBuffer);
            mainActivity.faceRecognizer = Dnn.readNetFromCaffe(bufferProto, bufferModel);
            Log.i(TAG, "MobileFaceNet initialized successfully!");
            return true;
        } catch (IOException e) {
            Log.e(TAG, "MobileFaceNet loaded failed" + e);
            Toast.makeText(mainActivity, "MobileFaceNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }
    }
}