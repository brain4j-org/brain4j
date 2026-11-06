package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.operations.impl.ActivationOperation;

/**
 * ONNX codec for {@link ActivationOperation}.
 */
public class SoftmaxOnnxCodec implements OnnxCodec<ActivationOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Softmax";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<ActivationOperation> targetClass() {
        return ActivationOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(ActivationOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public ActivationOperation decode(NodeProto node) {
        return new ActivationOperation(new Softmax());
    }
}
