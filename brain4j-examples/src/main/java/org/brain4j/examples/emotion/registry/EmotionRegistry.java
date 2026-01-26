package org.brain4j.examples.emotion.registry;

import org.brain4j.examples.emotion.PADEmotionalState;

import java.util.HashMap;
import java.util.Map;

public class EmotionRegistry {
    public static final Map<String, PADEmotionalState> EMOTIONS = new HashMap<>();

    static {
        EMOTIONS.put("Joy", new PADEmotionalState(0.8, 0.6, 0.5));
        EMOTIONS.put("Anger", new PADEmotionalState(-0.5, 0.7, 0.6));
        EMOTIONS.put("Sadness", new PADEmotionalState(-0.6, -0.3, -0.4));
        EMOTIONS.put("Fear", new PADEmotionalState(-0.7, 0.7, -0.6));
        EMOTIONS.put("Surprise", new PADEmotionalState(0.2, 0.8, 0.0));
        EMOTIONS.put("Disgust", new PADEmotionalState(-0.8, 0.3, 0.2));
        EMOTIONS.put("Calm", new PADEmotionalState(0.6, -0.5, 0.3));
        EMOTIONS.put("Anxiety", new PADEmotionalState(-0.4, 0.7, -0.3));
    }

    public static PADEmotionalState getEmotion(String name) {
        return EMOTIONS.get(name);
    }
}
