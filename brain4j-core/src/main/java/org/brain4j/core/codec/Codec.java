package org.brain4j.core.codec;

/**
 * Base codec contract.
 * All codecs expose a stable {@code type} discriminator and the target class.
 * JSON and ONNX specializations extend this interface.
 */
public interface Codec<T> {
    String type();
    Class<T> targetClass();
}
