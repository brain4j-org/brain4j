package org.brain4j.core.codec;

import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.Operation;

/**
 * ONNX specialization of {@link Codec} for {@link Operation}s.
 * The codec is responsible for translating an operation to/from ONNX {@code NodeProto} attributes.
 * <p>
 * The container ({@code OnnxFormat}) handles wiring of input/output tensor names;
 * the codec only adds/reads attributes and creates the operation instance.
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
