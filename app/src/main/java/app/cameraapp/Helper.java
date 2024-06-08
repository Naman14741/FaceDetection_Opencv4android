package app.cameraapp;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

public class Helper {
    private final MainActivity mainActivity;

    public enum ProcessMode {
        NORMAL,
        FACE_DETECTION,
        FACE_RECOGNITION,
        FACE_INSERTION
    }

    public Helper(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
    }

    @NonNull
    public byte[] getBytes(ImageProxy image) {
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
        return nv21;
    }

    public Bitmap getCenterCroppedBitmap(Bitmap srcBitmap, int targetWidth, int targetHeight) {
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

    public Bitmap combineBitmaps(Bitmap background, Bitmap overlay) {
        Bitmap combinedBitmap = Bitmap.createBitmap(background.getWidth(), background.getHeight(), background.getConfig());
        Canvas canvas = new Canvas(combinedBitmap);
        canvas.drawBitmap(background, new Matrix(), null);
        canvas.drawBitmap(overlay, new Matrix(), null);
        return combinedBitmap;
    }

    public void updateOverlay(Mat overlay) {
        Bitmap bitmap = Bitmap.createBitmap(overlay.cols(), overlay.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(overlay, bitmap);
        mainActivity.runOnUiThread(() -> mainActivity.overlayImageView.setImageBitmap(bitmap));
        overlay.release();
    }

    public void drawRoundedCornerRectangle(Mat overlay, float[] rectangle, Scalar color) {
        int thickness = 1;
        int x = Math.round(rectangle[0]);
        int y = Math.round(rectangle[1]);
        int w = Math.round(rectangle[2]);
        int h = Math.round(rectangle[3]);
        int r = Math.min(5, Math.min(w / 2, h / 2));

        Point topLeft = new Point(x + r, y + r);
        Point topRight = new Point(x + w - r, y + r);
        Point bottomRight = new Point(x + w - r, y + h - r);
        Point bottomLeft = new Point(x + r, y + h - r);

        Imgproc.ellipse(overlay, topLeft, new Size(r, r), 180, 0, 90, color, thickness);
        Imgproc.ellipse(overlay, topRight, new Size(r, r), 270, 0, 90, color, thickness);
        Imgproc.ellipse(overlay, bottomRight, new Size(r, r), 0, 0, 90, color, thickness);
        Imgproc.ellipse(overlay, bottomLeft, new Size(r, r), 90, 0, 90, color, thickness);

        Imgproc.line(overlay, new Point(x + r, y), new Point(x + w - r, y), color, thickness);
        Imgproc.line(overlay, new Point(x + w, y + r), new Point(x + w, y + h - r), color, thickness);
        Imgproc.line(overlay, new Point(x + w - r, y + h), new Point(x + r, y + h), color, thickness);
        Imgproc.line(overlay, new Point(x, y + h - r), new Point(x, y + r), color, thickness);
    }
}