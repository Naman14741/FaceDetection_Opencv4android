package app.cameraapp;

import android.content.Context;
import android.content.SharedPreferences;

public class Database {

    private static final String PREF_NAME = "face_data_pref";
    private static final String FACE_EMBEDDING_KEY = "face_embedding";
    private final SharedPreferences sharedPreferences;

    public Database(Context context) {
        this.sharedPreferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    public void saveFaceEmbedding(float[] embedding) {
        StringBuilder stringBuilder = new StringBuilder();
        for (float value : embedding) {
            stringBuilder.append(value).append(",");
        }
        sharedPreferences.edit().putString(FACE_EMBEDDING_KEY, stringBuilder.toString()).apply();
    }

    public float[] getFaceEmbedding() {
        String embeddingString = sharedPreferences.getString(FACE_EMBEDDING_KEY, null);
        if (embeddingString != null) {
            String[] parts = embeddingString.split(",");
            float[] embedding = new float[parts.length];
            for (int i = 0; i < parts.length; i++) {
                embedding[i] = Float.parseFloat(parts[i]);
            }
            return embedding;
        }
        return null;
    }

    public void clearFaceEmbedding() {
        sharedPreferences.edit().remove(FACE_EMBEDDING_KEY).apply();
    }

    public float cosineSimilarity(float[] embedding1, float[] embedding2) {
        float dotProduct = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int i = 0; i < embedding1.length; i++) {
            dotProduct += embedding1[i] * embedding2[i];
            norm1 += embedding1[i] * embedding1[i];
            norm2 += embedding2[i] * embedding2[i];
        }
        return dotProduct / (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
}