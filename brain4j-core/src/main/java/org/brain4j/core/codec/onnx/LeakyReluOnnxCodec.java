package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.activation.impl.LeakyReLU;
import org.brain4j.math.operations.impl.ActivationOperation;

/**
 * ONNX codec for {@link ActivationOperation}.
 */
public class LeakyReluOnnxCodec implements OnnxCodec<ActivationOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "LeakyRelu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<ActivationOperation> targetClass() {
        return ActivationOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(ActivationOperation op, NodeProto.Builder builder) {
        LeakyReLU act = (LeakyReLU) op.activation();
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("alpha")
            .setF((float) act.alpha())
            .setType(AttributeProto.AttributeType.FLOAT)
            .build());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ActivationOperation decode(NodeProto node) {
        double alpha = 0.01;

        for (AttributeProto attr : node.getAttributeList()) {
            if ("alpha".equals(attr.getName())) {
                alpha = attr.getF();
                break;
            }
        }

        return new ActivationOperation(new LeakyReLU(alpha));
    }
}
