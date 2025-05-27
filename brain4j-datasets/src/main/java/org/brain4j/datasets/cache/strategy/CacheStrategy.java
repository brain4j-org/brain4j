package org.brain4j.datasets.cache.strategy;

import java.nio.file.Path;

public interface CacheStrategy {
    boolean isValid(Path filePath);
    void clear(Path path);
}