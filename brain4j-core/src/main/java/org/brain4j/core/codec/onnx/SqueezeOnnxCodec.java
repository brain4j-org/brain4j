package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.SqueezeOperation;

public class SqueezeOnnxCodec implements OnnxCodec<SqueezeOperation> {
    @Override
    public String type() {
        return "Squeeze";
    }

    @Override
    public Class<SqueezeOperation> targetClass() {
        return SqueezeOperation.class;
    }

    @Override
    public void encode(SqueezeOperation op, NodeProto.Builder builder) {
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("axes")
            .setI(op.dim())
            .setType(AttributeProto.AttributeType.INT)
            .build());
    }

    @Override
    public SqueezeOperation decode(NodeProto node) {
        int dim = findInt(node, "axes", Integer.MAX_VALUE);
        if (dim == Integer.MAX_VALUE) {
            // try legacy "axis"
            dim = findInt(node, "axis", Integer.MAX_VALUE);
        }
        return new SqueezeOperation(dim);
    }

    private static int findInt(NodeProto node, String name, int def) {
        for (AttributeProto attr : node.getAttributeList()) {
            if (name.equals(attr.getName())) return (int) attr.getI();
        }
        if (node.getAttributeCount() > 0) return (int) node.getAttribute(0).getI();
        return def;
    }
}
