package app.cameraapp;

import android.content.ContentValues;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Display;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
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
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import app.cameraapp.Helper.ProcessMode;

public class MainActivity extends AppCompatActivity {
    private DisplayManager displayManager;
    private DisplayManager.DisplayListener displayListener;
    public float currentRotation = 0;
    private boolean rotated = false;
    private boolean isCameraStarted = false;
    public static ProcessMode processMode = ProcessMode.FACE_DETECTION;
    private static final String TAG = "FaceDetection";
    private PreviewView previewView;
    public ImageView overlayImageView;
    public FaceDetectorYN faceDetector;
    public Net faceRecognizer;
    public Database db;
    public final Helper helper = new Helper(this);
    private final Loader loader = new Loader(this);
    private final Visualizer visualizer = new Visualizer(this);
    private final Permissions permissions = new Permissions(this);
    int lensFacing = CameraSelector.LENS_FACING_BACK;
    private int displayRotation = 0;
    Camera camera;
    private Size mInputSize = null;
    private ExecutorService backgroundExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    public final ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(),
            this::onActivityResult
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        permissions.requestPermissions();
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        db = new Database(this);
        previewView = findViewById(R.id.previewView);
        ImageButton capture = findViewById(R.id.captureButton);
        ImageButton toggleFlash = findViewById(R.id.flashButton);
        ImageButton switchCamera = findViewById(R.id.switchCameraButton);
        ImageButton personAdd = findViewById(R.id.baselinepersonadd);
        ImageButton personOff = findViewById(R.id.roundpersonoff);
        ImageButton personSearch = findViewById(R.id.baselineperson);
        overlayImageView = findViewById(R.id.overlayImageView);

        boolean isOpenCVLoaded = loader.loadOpenCV();
        boolean isYuNetModelLoaded = loader.loadYuNetModel();
        boolean isMobileFaceNetModelLoaded = loader.loadMobileFaceNetModel();

        if (isOpenCVLoaded && isYuNetModelLoaded && isMobileFaceNetModelLoaded) {
            Toast.makeText(this, "All Models initialized successfully!", Toast.LENGTH_LONG).show();
            Log.i(TAG, "Models initialized successfully!");
        } else {
            Toast.makeText(this, "Models initialization failed!", Toast.LENGTH_LONG).show();
            Log.e(TAG, "Models initialization failed!");
        }

        capture.setOnClickListener(v -> captureImage());
        switchCamera.setOnClickListener(v -> switchCamera());
        toggleFlash.setOnClickListener(v -> toggleFlash());
        personAdd.setOnClickListener(v -> {
            processMode = ProcessMode.FACE_INSERTION;
            Toast.makeText(this, "Face insertion mode", Toast.LENGTH_SHORT).show();
        });
        personOff.setOnClickListener(v -> {
            ProcessMode previousMode = processMode;
            if (previousMode == ProcessMode.FACE_RECOGNITION) {
                Toast.makeText(this, "Face recognition turned off", Toast.LENGTH_SHORT).show();
                processMode = ProcessMode.FACE_DETECTION;
            } else if (previousMode == ProcessMode.FACE_DETECTION) {
                Toast.makeText(this, "Face detection turned off", Toast.LENGTH_SHORT).show();
                processMode = ProcessMode.NORMAL;
            } else {
                Toast.makeText(this, "Face recognition/detection already turned off", Toast.LENGTH_SHORT).show();
            }
        });
        personSearch.setOnClickListener(v -> {
            if (processMode != ProcessMode.FACE_RECOGNITION) {
                Toast.makeText(this, "Face recognition turned on", Toast.LENGTH_SHORT).show();
                processMode = ProcessMode.FACE_RECOGNITION;
            } else {
                Toast.makeText(this, "Face recognition already turned on", Toast.LENGTH_SHORT).show();
            }
        });

        displayManager = (DisplayManager) getSystemService(DISPLAY_SERVICE);
        Display defaultDisplay = displayManager.getDisplay(Display.DEFAULT_DISPLAY);
        if (defaultDisplay != null) {
            displayRotation = defaultDisplay.getRotation();
        }
        displayListener = new DisplayManager.DisplayListener() {
            @Override
            public void onDisplayAdded(int displayId) {}

            @Override
            public void onDisplayRemoved(int displayId) {}

            @Override
            public void onDisplayChanged(int displayId) {
                int difference = displayManager.getDisplay(displayId).getRotation() - displayRotation;
                displayRotation = displayManager.getDisplay(displayId).getRotation();
                if (difference == 2 || difference == -2 && !rotated) {
                    currentRotation = (currentRotation + 180) % 360;
                    rotated = true;
                    Log.i(TAG, "Display rotated by 180 degrees");
                    if (rotated) {
                        new Handler(Looper.getMainLooper()).postDelayed(() -> {
                            // Rotate the overlay after a delay
                            overlayImageView.setRotation(currentRotation);
                            rotated = false;
                        }, 2000);
                    }
                }
            }
        };
    }

    @Override
    public void onConfigurationChanged(@NonNull Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        adjustOverlayImageView();
    }

    private void adjustOverlayImageView() {
        // Adjust overlayImageView layout parameters to match full screen
        ViewGroup.LayoutParams params = overlayImageView.getLayoutParams();
        params.width = ViewGroup.LayoutParams.MATCH_PARENT;
        params.height = ViewGroup.LayoutParams.MATCH_PARENT;
        overlayImageView.setLayoutParams(params);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
        try {
            if (!backgroundExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                backgroundExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            backgroundExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        backgroundExecutor.shutdown();
        try {
            if (!backgroundExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                backgroundExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            backgroundExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        displayManager.unregisterDisplayListener(displayListener);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (backgroundExecutor.isShutdown()) {
            backgroundExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        }
        displayManager.registerDisplayListener(displayListener, null);
    }

    public void startCamera() {
        //android.util.Size targetResolution = new android.util.Size(4 * previewView.getWidth(), 4 * previewView.getHeight());
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
                        //.setResolutionStrategy(new ResolutionStrategy(targetResolution, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                        .build();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();

                imageAnalysis.setAnalyzer(backgroundExecutor, this::processImage);

                cameraProvider.unbindAll();

                camera = cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageCapture, imageAnalysis);
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                isCameraStarted = true;
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException(e);
            }

        }, ContextCompat.getMainExecutor(this));
    }

    public void captureImage() {
        if (camera != null) {
            // Create a bitmap of the PreviewView
            Bitmap previewBitmap = previewView.getBitmap();
            if (previewBitmap != null) {
                Bitmap combinedBitmap;

                if (overlayImageView.getDrawable() != null) {
                    Bitmap overlayBitmap = ((BitmapDrawable) overlayImageView.getDrawable()).getBitmap();
                    Bitmap croppedOverlayBitmap = helper.getCenterCroppedBitmap(overlayBitmap, previewBitmap.getWidth(), previewBitmap.getHeight());
                    combinedBitmap = helper.combineBitmaps(previewBitmap, croppedOverlayBitmap);
                } else {
                    combinedBitmap = previewBitmap;
                }

                // Convert the combined bitmap to a byte array
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                combinedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray = stream.toByteArray();

                // Create ContentValues for the new image
                ContentValues contentValues = new ContentValues();
                contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, "Image_" + System.currentTimeMillis());
                contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");

                // Get the content URI for the new image
                Uri imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);

                if (imageUri == null) {
                    Log.e(TAG, "Photo capture failed: imageUri is null");
                    return;
                }
                // Write the byte array to the content URI
                try (OutputStream os = getContentResolver().openOutputStream(imageUri)) {
                    if (os == null) {
                        Log.e(TAG, "Photo capture failed: OutputStream is null");
                        return;
                    }
                    os.write(byteArray);
                    // Display a success message
                    String msg = "Photo capture succeeded!";
                    Toast.makeText(getBaseContext(), msg, Toast.LENGTH_SHORT).show();
                    Log.d(TAG, msg);
                } catch (IOException e) {
                    Log.e(TAG, "Photo capture failed: ", e);
                }
            } else {
                Log.e(TAG, "Preview bitmap is null");
            }
        }
    }

    public void switchCamera() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK) ?
                CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        backgroundExecutor.shutdown();
        try {
            if (!backgroundExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                backgroundExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            backgroundExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        startCamera();
        backgroundExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
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
        byte[] nv21 = helper.getBytes(image);

        Mat yuvMat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
        yuvMat.put(0, 0, nv21);

        Mat rgbMat = new Mat();

        // Get image rotation
        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            switch (rotationDegrees) {
                case 90:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_COUNTERCLOCKWISE);
                    Core.flip(rgbMat, rgbMat, 1); // Flip horizontally
                    break;
                case 180:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_180);
                    Core.flip(rgbMat, rgbMat, 1); // Flip horizontally
                    break;
                case 270:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE);
                    Core.flip(rgbMat, rgbMat, 0); // Flip horizontally
                    break;
                default:
                    Core.flip(rgbMat, rgbMat, 1); // Flip horizontally for 0 degrees
                    break;
            }
        } else {
            switch (rotationDegrees) {
                case 90:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE);
                    break;
                case 180:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_180);
                    break;
                case 270:
                    Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_COUNTERCLOCKWISE);
                    break;
                default:
                    break;
            }
        }
        return rgbMat;
    }

    private void processImage(ImageProxy imageProxy) {
        if (!isCameraStarted || processMode == ProcessMode.NORMAL) {
            imageProxy.close();
            return;
        }

        // Convert ImageProxy to Mat
        Mat mat = yuvToRgb(imageProxy);
        // helper.updateOverlay(mat);

        if (mInputSize == null) {
            mInputSize = new Size(mat.cols(), mat.rows());
            faceDetector.setInputSize(mInputSize);
        }

        // Create a transparent overlay
        Mat overlay = Mat.zeros(mat.size(), CvType.CV_8UC4);

        // Resize mat to the input size of the model
        Imgproc.resize(mat, mat, mInputSize);

        // Convert color to BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2BGR);

        Mat faces = new Mat();
        faceDetector.setScoreThreshold(0.8f);
        faceDetector.detect(mat, faces);

        if (processMode == ProcessMode.FACE_INSERTION) {
            runOnUiThread(visualizer.visualizeFaceAdd(faces, mat));
        } else if (processMode == ProcessMode.FACE_DETECTION) {
            runOnUiThread(visualizer.visualizeFaceDetect(overlay, faces));
        } else if (processMode == ProcessMode.FACE_RECOGNITION) {
            runOnUiThread(visualizer.visualizeFaceRecognition(overlay, faces, mat));
        }

        // Update the overlay ImageView with the processed overlay
        helper.updateOverlay(overlay);

        // Release resources
        mat.release();
        faces.release();
        overlay.release();
        imageProxy.close();
    }

    private void onActivityResult(Map<String, Boolean> result) {
        boolean allPermissionsGranted = true;
        for (Map.Entry<String, Boolean> entry : result.entrySet()) {
            if (!entry.getValue()) {
                allPermissionsGranted = false;
                break;
            }
        }

        if (allPermissionsGranted) {
            startCamera();
        } else if (permissions.getDeniedPermissions().length > 0) {
            Toast.makeText(MainActivity.this, "You should accept all permissions to run this app", Toast.LENGTH_SHORT).show();
            requestPermissionLauncher.launch(permissions.getDeniedPermissions());
        }
    }
}
