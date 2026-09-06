package org.brain4j.transformers.api;

import java.nio.file.Path;

public record ModelFile(String name, Path path, long size, String format) {
}
