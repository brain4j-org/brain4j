package org.brain4j.transformers.cache.strategy;

import java.nio.file.Path;

public interface CacheStrategy {
    boolean isValid(Path filePath);
    void clear(Path path);
}