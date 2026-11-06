package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.operations.impl.AddOperation;

/**
 * ONNX codec for {@link AddOperation}.
 */
public class AddOnnxCodec implements OnnxCodec<AddOperation> {
    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "Add";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<AddOperation> targetClass() {
        return AddOperation.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void encode(AddOperation op, NodeProto.Builder builder) {}

    /**
     * {@inheritDoc}
     */
    @Override
    public AddOperation decode(NodeProto node) {
        return new AddOperation();
    }
}
