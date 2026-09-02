package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.MatMulOperation;

public class MatMulOnnxCodec implements OnnxCodec<MatMulOperation> {
    @Override
    public String type() {
        return "MatMul";
    }

    @Override
    public Class<MatMulOperation> targetClass() {
        return MatMulOperation.class;
    }

    @Override
    public void encode(MatMulOperation op, NodeProto.Builder builder) {}

    @Override
    public MatMulOperation decode(NodeProto node) {
        return new MatMulOperation();
    }
}
