package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.impl.ELU;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;

public class EluOnnxCodec implements OnnxCodec<ActivationOperation> {
    @Override
    public String type() {
        return "Elu";
    }

    @Override
    public Class<ActivationOperation> targetClass() {
        return ActivationOperation.class;
    }

    @Override
    public void encode(ActivationOperation op, NodeProto.Builder builder) {
        ELU elu = (ELU) op.activation();
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("alpha")
            .setF((float) elu.alpha())
            .setType(AttributeProto.AttributeType.FLOAT)
            .build());
    }

    @Override
    public ActivationOperation decode(NodeProto node) {
        double alpha = 1.0;
        for (AttributeProto attr : node.getAttributeList()) {
            if ("alpha".equals(attr.getName())) {
                alpha = attr.getF();
                break;
            }
        }
        return new ActivationOperation(new ELU(alpha));
    }
}
