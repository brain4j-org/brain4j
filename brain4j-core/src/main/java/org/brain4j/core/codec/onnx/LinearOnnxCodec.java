package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;

public class LinearOnnxCodec implements OnnxCodec<ActivationOperation> {
    @Override
    public String type() {
        return "Identity";
    }

    @Override
    public Class<ActivationOperation> targetClass() {
        return ActivationOperation.class;
    }

    @Override
    public void encode(ActivationOperation op, NodeProto.Builder builder) {}

    @Override
    public ActivationOperation decode(NodeProto node) {
        return new ActivationOperation(new Linear());
    }
}
