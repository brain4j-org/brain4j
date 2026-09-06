package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.importing.format.impl.OnnxFormat;
import org.brain4j.core.importing.io.OnnxIO;
import org.brain4j.math.tensor.Shape;

import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

public class OnnxOperationCodec implements JsonCodec<OnnxFormat.OnnxOperationLayer> {

    @Override
    public String type() {
        return "onnx_operation";
    }

    @Override
    public Class<OnnxFormat.OnnxOperationLayer> targetClass() {
        return OnnxFormat.OnnxOperationLayer.class;
    }

    @Override
    public void write(OnnxFormat.OnnxOperationLayer layer, ObjectNode out) {
        String opType = OnnxIO.encodeType(layer.operation());
        out.put("operation", opType != null ? opType : layer.operation().getClass().getSimpleName());
        out.put("inputs", String.join(",", layer.inputNames()));

        ObjectNode constants = MAPPER.createObjectNode();

        for (var entry : layer.constants().entrySet()) {
            ObjectNode tensorNode = constants.putObject(entry.getKey());
            tensorNode.put("shape", Shape.of(entry.getValue().shape()).toString());
        }

        out.set("constants", constants);
    }

    @Override
    public OnnxFormat.OnnxOperationLayer parse(JsonNode node) {
        throw new UnsupportedOperationException(
            "OnnxOperationLayer graphs must be loaded through the ONNX format, not JSON"
        );
    }
}
