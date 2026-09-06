package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.TransposeOperation;

public class TransposeOnnxCodec implements OnnxCodec<TransposeOperation> {
    @Override
    public String type() {
        return "Transpose";
    }

    @Override
    public Class<TransposeOperation> targetClass() {
        return TransposeOperation.class;
    }

    @Override
    public void encode(TransposeOperation op, NodeProto.Builder builder) {
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("perm_dim1")
            .setI(op.dim1())
            .setType(AttributeProto.AttributeType.INT)
            .build());
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("perm_dim2")
            .setI(op.dim2())
            .setType(AttributeProto.AttributeType.INT)
            .build());
    }

    @Override
    public TransposeOperation decode(NodeProto node) {
        int d1 = 0, d2 = 1;

        for (AttributeProto attr : node.getAttributeList()) {
            if ("perm_dim1".equals(attr.getName())) d1 = (int) attr.getI();
            if ("perm_dim2".equals(attr.getName())) d2 = (int) attr.getI();
        }

        if (node.getAttributeCount() >= 2 && d1 == 0 && d2 == 1) {
            d1 = (int) node.getAttribute(0).getI();
            d2 = (int) node.getAttribute(1).getI();
        }

        return new TransposeOperation(d1, d2);
    }
}
