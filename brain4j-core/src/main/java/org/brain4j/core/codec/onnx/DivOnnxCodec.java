package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.DivOperation;

public class DivOnnxCodec implements OnnxCodec<DivOperation> {
    @Override
    public String type() {
        return "Div";
    }

    @Override
    public Class<DivOperation> targetClass() {
        return DivOperation.class;
    }

    @Override
    public void encode(DivOperation op, NodeProto.Builder builder) {}

    @Override
    public DivOperation decode(NodeProto node) {
        return new DivOperation();
    }
}
