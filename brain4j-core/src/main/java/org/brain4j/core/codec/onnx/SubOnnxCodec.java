package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.SubOperation;

public class SubOnnxCodec implements OnnxCodec<SubOperation> {
    @Override
    public String type() {
        return "Sub";
    }

    @Override
    public Class<SubOperation> targetClass() {
        return SubOperation.class;
    }

    @Override
    public void encode(SubOperation op, NodeProto.Builder builder) {}

    @Override
    public SubOperation decode(NodeProto node) {
        return new SubOperation();
    }
}
