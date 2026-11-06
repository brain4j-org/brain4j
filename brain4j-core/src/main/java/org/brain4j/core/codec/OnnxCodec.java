package org.brain4j.core.codec;

import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.Operation;

/**
 * ONNX specialization of {@link Codec} for {@link Operation}s.
 * Translates operation attributes to and from ONNX {@code NodeProto} messages.
 *
 * @param <T> operation type
 */
public interface OnnxCodec<T extends Operation> extends Codec<T> {

    /**
     * Encode operation-specific attributes into the ONNX node builder.
     * The builder already has {@code opType} and {@code name} set.
     *
     * @param operation operation instance
     * @param builder   ONNX node builder to enrich with attributes
     */
    void encode(T operation, NodeProto.Builder builder);

    /**
     * Decode an operation from an ONNX node.
     *
     * @param node ONNX node
     * @return operation instance
     */
    T decode(NodeProto node);
}
