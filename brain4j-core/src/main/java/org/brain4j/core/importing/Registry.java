package org.brain4j.core.importing;

import org.brain4j.core.codec.Codec;

import java.util.HashMap;
import java.util.Map;

public class Registry<T> {
    
    private final Map<String, Codec<? extends T>> byType = new HashMap<>();
    private final Map<Class<?>, Codec<? extends T>> byClass = new HashMap<>();
    
    @SafeVarargs
    public final void put(Codec<? extends T>... codecs) {
        for (Codec<? extends T> codec : codecs) {
            byType.put(codec.type(), codec);
            byClass.put(codec.targetClass(), codec);
        }
    }
    
    @SuppressWarnings("unchecked")
    public <S extends T> Codec<T> get(Class<S> clazz) {
        return (Codec<T>) byClass.get(clazz);
    }
    
    @SuppressWarnings("unchecked")
    public Codec<T> get(String type) {
        return (Codec<T>) byType.get(type);
    }
}