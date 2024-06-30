package app.cameraapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Permissions {
    private static final Set<String> REQUIRED_PERMISSIONS = new HashSet<>(List.of());
    private final MainActivity mainActivity;
    public Permissions(MainActivity mainActivity) {
        this.mainActivity = mainActivity;
    }
    public enum StorageAccess {
        FULL,
        PARTIAL,
        DENIED
    }
    public static StorageAccess getStorageAccess(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED) {
            // Full access on Android 13+
            return StorageAccess.FULL;
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE &&
                ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) == PackageManager.PERMISSION_GRANTED) {
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

    public void requestPermissions() {
        List<String> permissionsToRequest = new ArrayList<>();

        // Camera permission
        if (ActivityCompat.checkSelfPermission(mainActivity, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.CAMERA);
        }

        // Storage permissions based on Android version
        StorageAccess storageAccess = getStorageAccess(mainActivity);
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

        REQUIRED_PERMISSIONS.addAll(permissionsToRequest);

        if (!permissionsToRequest.isEmpty()) {
            String[] permissionsArray = permissionsToRequest.toArray(new String[0]);
            mainActivity.requestPermissionLauncher.launch(permissionsArray);
        } else {
            mainActivity.startCamera();
        }
    }
    public String[] getDeniedPermissions() {
        List<String> deniedPermissions = new ArrayList<>();
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(mainActivity, permission) == PackageManager.PERMISSION_DENIED) {
                deniedPermissions.add(permission);
            }
        }
        return deniedPermissions.toArray(new String[0]);
    }
}