package org.brain4j.core;

import java.io.File;
import java.nio.file.Files;

import org.brain4j.core.importing.io.ModelIO;
import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.model.ModelLayout;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class SerializationTests {

    @Test
    void brainFormatRoundTripsModel() throws Exception {
        Sequential original = ModelLayout.of(Shape.of(2), new DenseLayer(3)).compile(123);
        File file = Files.createTempFile("brain4j-roundtrip-", ".zip").toFile();

        try {
            ModelIO.save(original, new BrainFormat(), file);
            Sequential loaded = ModelIO.load(new BrainFormat(), file);

            assertArrayEquals(original.layout().inputShape().dims(), loaded.layout().inputShape().dims());
            assertEquals(1, loaded.layers().size());
            assertEquals(DenseLayer.class, loaded.layers().getFirst().getClass());
            assertEquals(original.layers().getFirst().parameters().keySet(), loaded.layers().getFirst().parameters().keySet());

            for (String name : original.layers().getFirst().parameters().keySet()) {
                assertNotNull(loaded.layers().getFirst().parameters().get(name));
                assertArrayEquals(
                    original.layers().getFirst().parameters().get(name).shape(),
                    loaded.layers().getFirst().parameters().get(name).shape()
                );
            }

            assertArrayEquals(
                original.predict(Tensors.vector(0.25f, -0.5f).reshape(1, 2)).data(),
                loaded.predict(Tensors.vector(0.25f, -0.5f).reshape(1, 2)).data(),
                0.0f
            );
        } finally {
            Files.deleteIfExists(file.toPath());
        }
    }
}
