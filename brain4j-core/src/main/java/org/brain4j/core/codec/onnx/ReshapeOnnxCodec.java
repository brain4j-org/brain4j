package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.tensor.autograd.impl.ReshapeOperation;

import java.util.ArrayList;
import java.util.List;

public class ReshapeOnnxCodec implements OnnxCodec<ReshapeOperation> {
    @Override
    public String type() {
        return "Reshape";
    }

    @Override
    public Class<ReshapeOperation> targetClass() {
        return ReshapeOperation.class;
    }

    @Override
    public void encode(ReshapeOperation op, NodeProto.Builder builder) {
        for (int dim : op.newShape()) {
            builder.addAttribute(AttributeProto.newBuilder()
                .setName("shape")
                .setI(dim)
                .setType(AttributeProto.AttributeType.INTS)
                .build());
        }
    }

    @Override
    public ReshapeOperation decode(NodeProto node) {
        List<Integer> dims = new ArrayList<>();
        for (AttributeProto attr : node.getAttributeList()) {
            if ("shape".equals(attr.getName())) dims.add((int) attr.getI());
        }
        if (dims.isEmpty() && node.getAttributeCount() > 0) {
            for (AttributeProto attr : node.getAttributeList()) dims.add((int) attr.getI());
        }
        int[] shape = dims.stream().mapToInt(i -> i).toArray();
        return new ReshapeOperation(shape);
    }
}
