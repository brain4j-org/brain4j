package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.LayerNormOperation;

/**
 * ONNX codec for {@link LayerNormOperation}.
 */
public class LayerNormOnnxCodec implements OnnxCodec<LayerNormOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "LayerNormalization";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<LayerNormOperation> targetClass() {
        return LayerNormOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(LayerNormOperation op, NodeProto.Builder builder) {
        builder.addAttribute(AttributeProto.newBuilder()
            .setName("epsilon")
            .setF((float) op.epsilon())
            .setType(AttributeProto.AttributeType.FLOAT)
            .build());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LayerNormOperation decode(NodeProto node) {
        float eps = 1e-5f;

        for (AttributeProto attr : node.getAttributeList()) {
            if ("epsilon".equals(attr.getName())) {
                eps = attr.getF();
                break;
            }
        }

        if (node.getAttributeCount() > 0 && eps == 1e-5f) {
            // legacy: first attribute is epsilon

            try { eps = node.getAttribute(0).getF(); } catch (Exception ignored) {}
        }

        return new LayerNormOperation(eps);
    }
}
