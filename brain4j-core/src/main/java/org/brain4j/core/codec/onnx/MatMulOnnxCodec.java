package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.MatMulOperation;

/**
 * ONNX codec for {@link MatMulOperation}.
 */
public class MatMulOnnxCodec implements OnnxCodec<MatMulOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "MatMul";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<MatMulOperation> targetClass() {
        return MatMulOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(MatMulOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public MatMulOperation decode(NodeProto node) {
        return new MatMulOperation();
    }
}
