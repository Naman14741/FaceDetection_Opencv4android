package app.cameraapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.TorchState;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.core.Core;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "FaceDetection";
    ImageButton capture, toggleFlash, switchCamera;

    private PreviewView previewView;
    private ImageView overlayImageView;
    private FaceDetectorYN faceDetector;
    Net faceRecognizer;
    int lensFacing = CameraSelector.LENS_FACING_BACK;
    Camera camera;
    private Size mInputSize = null;
    private final ExecutorService backgroundExecutor = Executors.newSingleThreadExecutor();
    private final ActivityResultLauncher<String> activityResultLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), o -> {
        if (ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        capture = findViewById(R.id.captureButton);
        toggleFlash = findViewById(R.id.flashButton);
        switchCamera = findViewById(R.id.switchCameraButton);
        overlayImageView = findViewById(R.id.overlayImageView);

        boolean isOpenCVLoaded = loadOpenCV();
        boolean isYuNetModelLoaded = loadYuNetModel();
        boolean isMobileFaceNetModelLoaded = loadMobileFaceNetModel();

        if (isOpenCVLoaded && isYuNetModelLoaded && isMobileFaceNetModelLoaded) {
            Toast.makeText(this, "All Models initialized successfully!", Toast.LENGTH_LONG).show();
            Log.i(TAG, "Models initialized successfully!");
        } else {
            Toast.makeText(this, "Models initialization failed!", Toast.LENGTH_LONG).show();
            Log.e(TAG, "Models initialization failed!");
        }

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            activityResultLauncher.launch(Manifest.permission.CAMERA);
        }

        switchCamera.setOnClickListener(v -> switchCamera());
        toggleFlash.setOnClickListener(v -> toggleFlash());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
    }

    @Override
    protected void onPause() {
        super.onPause();
        backgroundExecutor.shutdown();
    }

    public void startCamera() {
        ListenableFuture<ProcessCameraProvider> listenableFuture = ProcessCameraProvider.getInstance(this);

        listenableFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = listenableFuture.get();

                Preview preview = new Preview.Builder().build();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build();

                ImageCapture imageCapture = new ImageCapture.Builder().build();

                // Camera resolution
                ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                        .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                        .build();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(backgroundExecutor, this::processImage);

                cameraProvider.unbindAll();

                camera = cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageCapture, imageAnalysis);
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException(e);
            }

        }, ContextCompat.getMainExecutor(this));
    }

    public void switchCamera() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK) ?
                CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        startCamera();
    }

    public void toggleFlash() {
        if (camera != null
                && camera.getCameraInfo().hasFlashUnit()
                && camera.getCameraInfo().getTorchState().getValue() != null)
        {
            boolean isFlashOn = (camera.getCameraInfo().getTorchState().getValue() == TorchState.ON);
            camera.getCameraControl().enableTorch(!isFlashOn);
        }
    }

    private Mat yuvToRgb(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        Mat yuvMat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
        yuvMat.put(0, 0, nv21);

        Mat rgbMat = new Mat();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_I420);
        Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE);
        return rgbMat;
    }



    private boolean loadOpenCV() {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
            return true;
        } else {
            Log.e(TAG, "OpenCV loaded failed!");
            Toast.makeText(this, "OpenCV was not found!", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    private boolean loadYuNetModel() {
        try (InputStream is = this.getResources().openRawResource(R.raw.face_detection_yunet_2023mar)) {
            int size = is.available();
            byte[] buffer = new byte[size];
            //noinspection ResultOfMethodCallIgnored
            is.read(buffer);
            MatOfByte mModelBuffer = new MatOfByte(buffer);
            MatOfByte mConfigBuffer = new MatOfByte();
            faceDetector = FaceDetectorYN.create("onnx", mModelBuffer, mConfigBuffer, new Size(320, 320));

            Log.i(TAG, "YuNet initialized successfully!");
            return true;
        } catch (IOException e) {
            Log.e(TAG, "YuNet loaded failed" + e);
            Toast.makeText(this, "YuNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    private boolean loadMobileFaceNetModel() {
        try (InputStream protoIs = this.getResources().openRawResource(R.raw.mobilefacenet);
             InputStream modelIs = this.getResources().openRawResource(R.raw.mobilefacenet_caffemodel)) {
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
            faceRecognizer = Dnn.readNetFromCaffe(bufferProto, bufferModel);
            Log.i(TAG, "MobileFaceNet initialized successfully!");
            return true;
        } catch (IOException e) {
            Log.e(TAG, "MobileFaceNet loaded failed" + e);
            Toast.makeText(this, "MobileFaceNet model was not found", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    private void processImage(ImageProxy imageProxy) {
        Mat mat = yuvToRgb(imageProxy);

        if (mInputSize == null) {
            mInputSize = new Size(mat.cols(),mat.rows());
            faceDetector.setInputSize(mInputSize);
        }

        // Resize mat to the input size of the model
        Imgproc.resize(mat, mat, mInputSize);

        // Convert color to BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2BGR);

        Mat faces = new Mat();
        faceDetector.setScoreThreshold(0.8f);
        faceDetector.detect(mat, faces);

        // Create a transparent overlay
        Mat overlay = Mat.zeros(mat.size(), CvType.CV_8UC4);

        // Draw bounding boxes on the transparent overlay
        runOnUiThread(visualize(overlay, faces));

        // Update the overlay ImageView with the processed overlay
        updateOverlay(overlay);

        mat.release();
        faces.release();
        overlay.release();
        imageProxy.close();
    }

    // Draw bounding boxes on the transparent overlay
    private Runnable visualize(Mat overlay, Mat faces) {
        int thickness = 1;
        float[] faceData = new float[faces.cols() * faces.channels()];
    
        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);
            Imgproc.rectangle(overlay, new Rect(Math.round(faceData[0]), Math.round(faceData[1]),
                            Math.round(faceData[2]), Math.round(faceData[3])),
                    new Scalar(0, 255, 0, 255), thickness); // Using RGBA for transparency
        }
        return null;
    }

    private void updateOverlay(Mat overlay) {
        Bitmap bitmap = Bitmap.createBitmap(overlay.cols(), overlay.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(overlay, bitmap);
        runOnUiThread(() -> overlayImageView.setImageBitmap(bitmap));
        overlay.release();
    }
}