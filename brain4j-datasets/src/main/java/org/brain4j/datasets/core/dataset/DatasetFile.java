package org.brain4j.datasets.core.dataset;

import java.nio.file.Path;

public record DatasetFile(String name, Path path, long size, String format) {
}