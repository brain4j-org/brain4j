package org.brain4j.core.serializing;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Interface used to allow serialization and deserialization of objects in binary.
 */
public interface BinarySerializable {
    /**
     * Serializes the object to the output stream.
     * @param stream The output stream
     * @throws Exception If an exception occurs during the serialization
     */
    void serialize(DataOutputStream stream) throws Exception;

    /**
     * Deserializes the object from the input stream.
     * @param stream The input stream
     * @throws Exception If an exception occurs during the deserialization
     */
    void deserialize(DataInputStream stream) throws Exception;
}
