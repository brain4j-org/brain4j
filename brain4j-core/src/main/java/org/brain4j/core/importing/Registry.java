package org.brain4j.core.importing;

import org.brain4j.core.codec.Codec;

import java.util.HashMap;
import java.util.Map;

public class Registry<T> {
    
    private final Map<String, Codec<? extends T>> byType = new HashMap<>();
    private final Map<Class<?>, Codec<? extends T>> byClass = new HashMap<>();
    
    public void put(Codec<? extends T> codec) {
        byType.put(codec.type(), codec);
        byClass.put(codec.targetClass(), codec);
    }
    
    @SuppressWarnings("unchecked")
    public <S extends T> Codec<S> get(Class<S> clazz) {
        return (Codec<S>) byClass.get(clazz);
    }
    
    @SuppressWarnings("unchecked")
    public <S extends T> Codec<S> get(String type) {
        return (Codec<S>) byType.get(type);
    }
}