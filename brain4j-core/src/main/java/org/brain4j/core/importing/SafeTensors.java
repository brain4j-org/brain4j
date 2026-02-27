package org.brain4j.core.importing;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

public class SafeTensors {

    public static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    public static byte[] create(Map<String, Tensor> weights) {
        try {
            ObjectNode header = MAPPER.createObjectNode();
            int offset = 0;

            for (Map.Entry<String, Tensor> entry : weights.entrySet()) {
                String name = entry.getKey();
                Tensor weight = entry.getValue();

                int begin = offset;
                offset += weight.elements() * 4;
                int end = offset;

                ArrayNode offsets = MAPPER.createArrayNode();
                offsets.add(begin);
                offsets.add(end);

                ArrayNode shape = MAPPER.createArrayNode();
                for (int dimension : weight.shape()) {
                    shape.add(dimension);
                }

                ObjectNode tensor = MAPPER.createObjectNode();
                tensor.put("dtype", "f32");
                tensor.set("shape", shape);
                tensor.set("data_offsets", offsets);

                header.set(name, tensor);
            }

            byte[] headerJson = MAPPER.writeValueAsBytes(header);

            try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {

                ByteBuffer buffer = ByteBuffer
                    .allocate(8)
                    .order(NATIVE_ORDER)
                    .putLong(headerJson.length);

                stream.write(buffer.array());
                stream.write(headerJson);

                for (Tensor tensor : weights.values()) {
                    stream.write(tensor.toByteArray());
                }

                return stream.toByteArray();
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to create safetensors", e);
        }
    }

    public static Map<String, Tensor> load(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        return load(data);
    }

    public static Map<String, Tensor> load(byte[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(NATIVE_ORDER);
        return load(buffer);
    }

    private static Map<String, Tensor> load(ByteBuffer buffer) throws IOException {
        buffer.order(NATIVE_ORDER);

        if (buffer.remaining() < 8) {
            throw new IOException("Invalid safetensors buffer");
        }

        long headerLengthLong = buffer.getLong();

        if (headerLengthLong > Integer.MAX_VALUE) {
            throw new IOException("Header too large (>2GB)");
        }

        int headerLength = (int) headerLengthLong;

        if (buffer.remaining() < headerLength) {
            throw new IOException("Unexpected EOF while reading header");
        }

        byte[] headerBytes = new byte[headerLength];
        buffer.get(headerBytes);

        JsonNode header = MAPPER.readTree(headerBytes);

        long baseDataOffset = 8L + headerLength;
        Map<String, Tensor> weights = new HashMap<>();

        header.fields().forEachRemaining(entry -> {
            String name = entry.getKey();

            if (name.equals("__metadata__")) return;

            JsonNode info = entry.getValue();

            JsonNode shapeArray = info.get("shape");
            if (shapeArray == null || !shapeArray.isArray()) {
                throw Commons.illegalArgument("Invalid or missing shape for tensor: " + name);
            }

            int[] shape = new int[shapeArray.size()];
            int elements = 1;

            for (int i = 0; i < shape.length; i++) {
                JsonNode dim = shapeArray.get(i);

                if (!dim.isInt()) {
                    throw Commons.illegalArgument("Invalid shape dimension in tensor: " + name);
                }

                shape[i] = dim.intValue();
                elements *= shape[i];
            }

            JsonNode offsets = info.has("offsets")
                ? info.get("offsets")
                : info.get("data_offsets");

            if (offsets == null || !offsets.isArray() || offsets.size() != 2) {
                throw Commons.illegalArgument("Invalid offsets for tensor: " + name);
            }

            long start = offsets.get(0).longValue();
            long end = offsets.get(1).longValue();

            int byteLength = Math.toIntExact(end - start);

            int pos = Math.toIntExact(baseDataOffset + start);

            ByteBuffer slice = buffer.duplicate();
            slice.position(pos);
            slice.limit(pos + byteLength);
            slice = slice.slice().order(ByteOrder.LITTLE_ENDIAN);

            float[] values = new float[elements];
            slice.asFloatBuffer().get(values);

            weights.put(name, Tensors.create(shape, values));
        });

        return weights;
    }
}