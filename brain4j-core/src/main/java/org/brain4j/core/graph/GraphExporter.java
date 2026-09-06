package org.brain4j.core.graph;

import org.brain4j.core.model.impl.Graph;

import java.io.*;
import java.nio.charset.StandardCharsets;

public interface GraphExporter {
    String serialize(Graph model);

    default void export(Graph model, File outFile) {
        try (FileOutputStream fileOutputStream = new FileOutputStream(outFile)) {
            try (BufferedOutputStream outputStream = new BufferedOutputStream(fileOutputStream)) {
                String out = serialize(model);
                outputStream.write(out.getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
