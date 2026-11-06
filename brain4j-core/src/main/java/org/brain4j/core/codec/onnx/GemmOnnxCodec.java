package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.GemmOperation;

/**
 * ONNX codec for {@link GemmOperation}.
 */
public class GemmOnnxCodec implements OnnxCodec<GemmOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Gemm";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<GemmOperation> targetClass() {
        return GemmOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(GemmOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public GemmOperation decode(NodeProto node) {
        return new GemmOperation();
    }
}
