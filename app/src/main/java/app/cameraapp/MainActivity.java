package app.cameraapp;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
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
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    public ProcessMode processMode = ProcessMode.NORMAL;
    private static final String TAG = "FaceDetection";
    private PreviewView previewView;
    private ImageView overlayImageView;
    private FaceDetectorYN faceDetector;
    private Net faceRecognizer;
    private Database db;
    int lensFacing = CameraSelector.LENS_FACING_BACK;
    Camera camera;
    private Size mInputSize = null;
    private final ExecutorService backgroundExecutor = Executors.newSingleThreadExecutor();
    private final ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(
        new ActivityResultContracts.RequestMultiplePermissions(),
        result -> {
            boolean allPermissionsGranted = true;
            for (Map.Entry<String, Boolean> entry : result.entrySet()) {
                if (!entry.getValue()) {
                    allPermissionsGranted = false;
                    break;
                }
            }
            if (allPermissionsGranted) {
                startCamera();
            } else {
                Toast.makeText(MainActivity.this, "Permissions denied", Toast.LENGTH_SHORT).show();
            }
        }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
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

        requestPermissions();

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
    }

    private void requestPermissions() {
        List<String> permissionsToRequest = new ArrayList<>();

        // Camera permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.CAMERA);
        }

        // Storage permissions based on Android version
        StorageAccess storageAccess = getStorageAccess(this);
        if (storageAccess == StorageAccess.DENIED) {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            } else {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    permissionsToRequest.add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED);
                }
            }
        }

        if (!permissionsToRequest.isEmpty()) {
            String[] permissionsArray = permissionsToRequest.toArray(new String[0]);
            requestPermissionLauncher.launch(permissionsArray);
        } else {
            startCamera();
        }
    }

    public enum StorageAccess {
        FULL,
        PARTIAL,
        DENIED
    }

    public static StorageAccess getStorageAccess(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED) {
            // Full access on Android 13+
            return StorageAccess.FULL;
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE &&
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) == PackageManager.PERMISSION_GRANTED) {
            // Partial access on Android 14+
            return StorageAccess.PARTIAL;
        } else if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            // Full access up to Android 12
            return StorageAccess.FULL;
        } else {
            // Access denied
            return StorageAccess.DENIED;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // For debugging purposes
        db.clearFaceEmbedding();
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

    public void captureImage() {
        if (camera != null) {
            // Create a bitmap of the PreviewView
            Bitmap previewBitmap = previewView.getBitmap();
            if (previewBitmap != null) {
                Bitmap combinedBitmap;

                if (overlayImageView.getDrawable() != null) {
                    Bitmap overlayBitmap = ((BitmapDrawable) overlayImageView.getDrawable()).getBitmap();
                    Bitmap croppedOverlayBitmap = getCenterCroppedBitmap(overlayBitmap, previewBitmap.getWidth(), previewBitmap.getHeight());
                    combinedBitmap = combineBitmaps(previewBitmap, croppedOverlayBitmap);
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

    private Bitmap getCenterCroppedBitmap(Bitmap srcBitmap, int targetWidth, int targetHeight) {
        // Calculate source rectangle
        int srcWidth = srcBitmap.getWidth();
        int srcHeight = srcBitmap.getHeight();
        float srcAspectRatio = (float) srcWidth / srcHeight;
        float targetAspectRatio = (float) targetWidth / targetHeight;

        int cropWidth, cropHeight;
        int cropX, cropY;

        if (srcAspectRatio > targetAspectRatio) {
            cropHeight = srcHeight;
            cropWidth = (int) (cropHeight * targetAspectRatio);
            cropX = (srcWidth - cropWidth) / 2;
            cropY = 0;
        } else {
            cropWidth = srcWidth;
            cropHeight = (int) (cropWidth / targetAspectRatio);
            cropX = 0;
            cropY = (srcHeight - cropHeight) / 2;
        }

        // Create a cropped bitmap
        Bitmap croppedBitmap = Bitmap.createBitmap(srcBitmap, cropX, cropY, cropWidth, cropHeight);

        // Scale the cropped bitmap to the target size
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, targetWidth, targetHeight, true);

        // Recycle the cropped bitmap to save memory
        croppedBitmap.recycle();

        return scaledBitmap;
    }

    private Bitmap combineBitmaps(Bitmap background, Bitmap overlay) {
        Bitmap combinedBitmap = Bitmap.createBitmap(background.getWidth(), background.getHeight(), background.getConfig());
        Canvas canvas = new Canvas(combinedBitmap);
        canvas.drawBitmap(background, new Matrix(), null);
        canvas.drawBitmap(overlay, new Matrix(), null);
        return combinedBitmap;
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

        byte[] i420 = new byte[ySize + uSize + vSize];

        // U and V are swapped
        yBuffer.get(i420, 0, ySize);
        vBuffer.get(i420, ySize, vSize);
        uBuffer.get(i420, ySize + vSize, uSize);

        Mat yuvMat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
        yuvMat.put(0, 0, i420);

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
            faceDetector.setScoreThreshold(0.8f);
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

    public enum ProcessMode {
        NORMAL,
        FACE_DETECTION,
        FACE_RECOGNITION,
        FACE_INSERTION
    }

    private void processImage(ImageProxy imageProxy) {
        if (processMode == ProcessMode.NORMAL) {
            imageProxy.close();
            return;
        }

        // Convert ImageProxy to Mat
        Mat mat = yuvToRgb(imageProxy);

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
            runOnUiThread(visualizeFaceAdd(faces, mat));
        } else if (processMode == ProcessMode.FACE_DETECTION) {
            runOnUiThread(visualizeFaceDetect(overlay, faces));
        } else if (processMode == ProcessMode.FACE_RECOGNITION) {
            runOnUiThread(visualizeFaceRecognition(overlay, faces, mat));
        }

        // Update the overlay ImageView with the processed overlay
        updateOverlay(overlay);

        // Release resources
        mat.release();
        faces.release();
        overlay.release();
        imageProxy.close();
    }

    private float[] extractFaceEmbedding(Mat face) {
        Mat blob = Dnn.blobFromImage(face, 1.0 / 128.0, new Size(112, 112), new Scalar(127.5, 127.5, 127.5), true, false);
        faceRecognizer.setInput(blob);
        Mat embedding = faceRecognizer.forward();

        float[] embeddingData = new float[(int) embedding.total()];
        embedding.get(0, 0, embeddingData);
        Log.i(TAG, "Face embedding: " + Arrays.toString(embeddingData));

        // Release resources
        blob.release();
        embedding.release();

        return embeddingData;
    }

    // Draw bounding boxes on the transparent overlay
    private Runnable visualizeFaceDetect(Mat overlay, Mat faces) {
        int thickness = 2;
        float[] faceData = new float[faces.cols() * faces.channels()];

        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);
            Imgproc.rectangle(overlay, new Rect(Math.round(faceData[0]), Math.round(faceData[1]),
                            Math.round(faceData[2]), Math.round(faceData[3])),
                    new Scalar(0, 255, 0, 255), thickness); // Using RGBA for transparency
        }
        return null;
    }
    private Runnable visualizeFaceAdd(Mat faces, Mat mat) {
        int numFaces = faces.rows();
        if (numFaces != 0) {
            float[] faceData = new float[faces.cols() * faces.channels()];

            for (int i = 0; i < numFaces; i++) {
                faces.get(i, 0, faceData);
                Rect faceRect = new Rect(Math.round(faceData[0]), Math.round(faceData[1]),
                        Math.round(faceData[2]), Math.round(faceData[3]));

                // Check if the face rectangle is within the image boundaries
                if (faceRect.x >= 0 && faceRect.y >= 0 && faceRect.x + faceRect.width <= mat.cols() && faceRect.y + faceRect.height <= mat.rows()) {
                    // Extract face embedding
                    Mat face = new Mat(mat, faceRect);
                    float[] embedding = extractFaceEmbedding(face);
                    float[] dbEmbedding = db.getFaceEmbedding();

                    if (dbEmbedding == null) {
                        if (!isDialogShown) {
                            isDialogShown = true;
                            runOnUiThread(() -> {
                                // Show a dialog asking the user if they want to save the new face
                                new AlertDialog.Builder(this)
                                        .setTitle("No Saved Face")
                                        .setMessage("No saved face found. Would you like to save this new face?")
                                        .setPositiveButton("Yes", (dialog, which) -> {
                                            // Save the face embedding to the database
                                            db.saveFaceEmbedding(embedding);
                                            Toast.makeText(this, "New face saved!", Toast.LENGTH_SHORT).show();
                                            Log.i(TAG, "Face embedding saved to database");
                                        })
                                        .setNegativeButton("No", (dialog, which) -> {
                                            // Do nothing
                                        })
                                        .show();
                            });
                        }
                    } else {
                        runOnUiThread(() -> {
                            // Show a dialog asking the user if they want to save the new face
                            new AlertDialog.Builder(this)
                                    .setTitle("Saved Face Found")
                                    .setMessage("Saved face found. Would you like to save this new face")
                                    .setPositiveButton("Yes", (dialog, which) -> {
                                        // Save the face embedding to the database
                                        db.saveFaceEmbedding(embedding);
                                        Toast.makeText(this, "New face saved!", Toast.LENGTH_SHORT).show();
                                        Log.i(TAG, "Face embedding saved to database");
                                    })
                                    .setNegativeButton("No", (dialog, which) -> {
                                        // Do nothing
                                    })
                                    .show();
                        });
                    }
                    face.release();
                }
            }
        }
        processMode = ProcessMode.FACE_RECOGNITION;
        return null;
    }
    private boolean isDialogShown = false;
    private Runnable visualizeFaceRecognition(Mat overlay, Mat faces, Mat mat) {
        int numFaces = faces.rows();
        if (numFaces != 0) {
            int thickness = 2;
            float[] faceData = new float[faces.cols() * faces.channels()];

            for (int i = 0; i < numFaces; i++) {
                faces.get(i, 0, faceData);
                Rect faceRect = new Rect(Math.round(faceData[0]), Math.round(faceData[1]),
                        Math.round(faceData[2]), Math.round(faceData[3]));

                // Check if the face rectangle is within the image boundaries
                if (faceRect.x >= 0 && faceRect.y >= 0 && faceRect.x + faceRect.width <= mat.cols() && faceRect.y + faceRect.height <= mat.rows()) {
                    // Extract face embedding
                    Mat face = new Mat(mat, faceRect);
                    float[] embedding = extractFaceEmbedding(face);
                    float[] dbEmbedding = db.getFaceEmbedding();

                    String text;
                    if (dbEmbedding == null) {
                        processMode = ProcessMode.FACE_INSERTION;
                    } else {
                        // Calculate cosine similarity
                        float similarity = db.cosineSimilarity(embedding, dbEmbedding);
                        Scalar color;
                        if (similarity > 0.8) {
                            text = "Matched";
                            color = new Scalar(0, 255, 0, 255);
                            Imgproc.rectangle(overlay, faceRect, new Scalar(0, 255, 0, 255), thickness); // Using RGBA for transparency

                        } else {
                            text = "Not Matched";
                            color = new Scalar(255, 0, 0, 255);
                            Imgproc.rectangle(overlay, faceRect, new Scalar(255, 0, 0, 255), thickness); // Using RGBA for transparency

                        }
                        Imgproc.putText(overlay, text, new Point(faceRect.x, faceRect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness);
                        Log.i(TAG, "Cosine similarity: " + similarity);
                        Log.i(TAG, "Embedding: " + Arrays.toString(embedding));
                        Log.i(TAG, "Database embedding: " + Arrays.toString(dbEmbedding));
                    }
                    face.release();
                }
            }
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