package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.SubOperation;

/**
 * ONNX codec for {@link SubOperation}.
 */
public class SubOnnxCodec implements OnnxCodec<SubOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Sub";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<SubOperation> targetClass() {
        return SubOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(SubOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public SubOperation decode(NodeProto node) {
        return new SubOperation();
    }
}
