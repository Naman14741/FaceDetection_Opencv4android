package app.cameraapp;

import android.app.AlertDialog;
import android.util.Log;
import android.widget.Toast;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

public class Visualizer {
    private final MainActivity mainActivity;
    private static final String TAG = "Visualizer";
    public Visualizer(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
    }

    private float[] extractFaceEmbedding(Mat face) {
        Mat blob = Dnn.blobFromImage(face, 1.0 / 255.0, new Size(112, 112), new Scalar(127.5, 127.5, 127.5), true, false);
        mainActivity.faceRecognizer.setInput(blob);
        Mat embedding = mainActivity.faceRecognizer.forward();

        float[] embeddingData = new float[(int) embedding.total()];
        embedding.get(0, 0, embeddingData);
        Log.i(TAG, "Face embedding: " + Arrays.toString(embeddingData));

        // Release resources
        blob.release();
        embedding.release();

        return embeddingData;
    }
    public Runnable visualizeFaceDetect(Mat overlay, Mat faces) {
        float[] faceData = new float[faces.cols() * faces.channels()];

        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);
            mainActivity.helper.drawRoundedCornerRectangle(overlay, faceData, new Scalar(0, 255, 0, 255));
        }
        return null;
    }
    public Runnable visualizeFaceAdd(Mat faces, Mat mat) {
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
                    float[] dbEmbedding = mainActivity.db.getFaceEmbedding();

                    if (dbEmbedding == null) {
                        if (!isDialogShown) {
                            isDialogShown = true;
                            mainActivity.runOnUiThread(() -> {
                                // Show a dialog asking the user if they want to save the new face
                                new AlertDialog.Builder(mainActivity)
                                        .setTitle("No Saved Face")
                                        .setMessage("No saved face found. Would you like to save this new face?")
                                        .setPositiveButton("Yes", (dialog, which) -> {
                                            // Save the face embedding to the database
                                            mainActivity.db.saveFaceEmbedding(embedding);
                                            Toast.makeText(mainActivity, "New face saved!", Toast.LENGTH_SHORT).show();
                                            Log.i(TAG, "Face embedding saved to database");
                                        })
                                        .setNegativeButton("No", (dialog, which) -> {
                                            // Do nothing
                                        })
                                        .show();
                            });
                        }
                    } else {
                        mainActivity.runOnUiThread(() -> {
                            // Show a dialog asking the user if they want to save the new face
                            new AlertDialog.Builder(mainActivity)
                                    .setTitle("Saved Face Found")
                                    .setMessage("Saved face found. Would you like to save this new face")
                                    .setPositiveButton("Yes", (dialog, which) -> {
                                        // Save the face embedding to the database
                                        mainActivity.db.saveFaceEmbedding(embedding);
                                        Toast.makeText(mainActivity, "New face saved!", Toast.LENGTH_SHORT).show();
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
        MainActivity.processMode = Helper.ProcessMode.FACE_RECOGNITION;
        return null;
    }
    private static boolean isDialogShown = false;
    public Runnable visualizeFaceRecognition(Mat overlay, Mat faces, Mat mat) {
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
                    float[] dbEmbedding = mainActivity.db.getFaceEmbedding();

                    String text;
                    if (dbEmbedding == null) {
                        MainActivity.processMode = Helper.ProcessMode.FACE_INSERTION;
                    } else {
                        // Calculate cosine similarity
                        float similarity = mainActivity.db.cosineSimilarity(embedding, dbEmbedding);
                        Scalar color;
                        if (similarity > 0.8) {
                            text = "Matched";
                            color = new Scalar(0, 255, 0, 255);
                        } else {
                            text = "Not Matched";
                            color = new Scalar(255, 0, 0, 255);
                        }
                        mainActivity.helper.drawRoundedCornerRectangle(overlay, faceData, color);
                        Imgproc.putText(overlay, text, new Point(faceRect.x, faceRect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
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
}