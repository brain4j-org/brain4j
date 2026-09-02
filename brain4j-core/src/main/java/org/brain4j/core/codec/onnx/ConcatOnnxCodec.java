package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.ConcatOperation;

public class ConcatOnnxCodec implements OnnxCodec<ConcatOperation> {

    @Override
    public String type() {
        return "Concat";
    }

    @Override
    public Class<ConcatOperation> targetClass() {
        return ConcatOperation.class;
    }

    @Override
    public void encode(ConcatOperation op, NodeProto.Builder builder) {
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("axis")
            .setI(op.dimension())
            .setType(AttributeProto.AttributeType.INT)
            .build());
    }

    @Override
    public ConcatOperation decode(NodeProto node) {
        int axis = findInt(node, "axis", 0);
        return new ConcatOperation(axis);
    }

    private static int findInt(NodeProto node, String name, int def) {
        for (AttributeProto attr : node.getAttributeList()) {
            if (name.equals(attr.getName())) return (int) attr.getI();
        }
        // fallback to first attribute for legacy files
        if (node.getAttributeCount() > 0) return (int) node.getAttribute(0).getI();
        return def;
    }
}
