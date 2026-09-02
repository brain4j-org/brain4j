package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.AddOperation;

public class AddOnnxCodec implements OnnxCodec<AddOperation> {
    @Override
    public String type() {
        return "Add";
    }

    @Override
    public Class<AddOperation> targetClass() {
        return AddOperation.class;
    }

    @Override
    public void encode(AddOperation op, NodeProto.Builder builder) {}

    @Override
    public AddOperation decode(NodeProto node) {
        return new AddOperation();
    }
}
