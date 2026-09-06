package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.MulOperation;

public class MulOnnxCodec implements OnnxCodec<MulOperation> {
    @Override
    public String type() {
        return "Mul";
    }

    @Override
    public Class<MulOperation> targetClass() {
        return MulOperation.class;
    }

    @Override
    public void encode(MulOperation op, NodeProto.Builder builder) {}

    @Override
    public MulOperation decode(NodeProto node) {
        return new MulOperation();
    }
}
