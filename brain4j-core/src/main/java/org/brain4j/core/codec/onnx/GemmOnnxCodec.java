package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.GemmOperation;

public class GemmOnnxCodec implements OnnxCodec<GemmOperation> {
    @Override
    public String type() {
        return "Gemm";
    }

    @Override
    public Class<GemmOperation> targetClass() {
        return GemmOperation.class;
    }

    @Override
    public void encode(GemmOperation op, NodeProto.Builder builder) {}

    @Override
    public GemmOperation decode(NodeProto node) {
        return new GemmOperation();
    }
}
