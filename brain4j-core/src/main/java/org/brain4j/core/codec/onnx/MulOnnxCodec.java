package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.MulOperation;

/**
 * ONNX codec for {@link MulOperation}.
 */
public class MulOnnxCodec implements OnnxCodec<MulOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Mul";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<MulOperation> targetClass() {
        return MulOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(MulOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public MulOperation decode(NodeProto node) {
        return new MulOperation();
    }
}
