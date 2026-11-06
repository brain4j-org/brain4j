package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.DivOperation;

/**
 * ONNX codec for {@link DivOperation}.
 */
public class DivOnnxCodec implements OnnxCodec<DivOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Div";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<DivOperation> targetClass() {
        return DivOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(DivOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public DivOperation decode(NodeProto node) {
        return new DivOperation();
    }
}
