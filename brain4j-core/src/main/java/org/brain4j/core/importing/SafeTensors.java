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
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static java.nio.channels.FileChannel.MapMode.READ_ONLY;
import static java.nio.file.StandardOpenOption.READ;

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
        try (FileChannel channel = FileChannel.open(path, READ)) {
            return load(channel);
        }
    }

    public static Map<String, Tensor> load(byte[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(NATIVE_ORDER);
        return load(buffer);
    }

    private static Map<String, Tensor> load(FileChannel channel) throws IOException {
        long fileSize = channel.size();

        if (fileSize < 8) {
            throw new IOException("Invalid safetensors buffer");
        }

        ByteBuffer headerSizeBuffer = ByteBuffer.allocate(8).order(NATIVE_ORDER);
        readFully(channel, headerSizeBuffer, 0);
        headerSizeBuffer.flip();

        long headerLengthLong = headerSizeBuffer.getLong();

        if (headerLengthLong > Integer.MAX_VALUE) {
            throw new IOException("Header too large (>2GB)");
        }

        int headerLength = (int) headerLengthLong;
        long baseDataOffset = 8L + headerLength;

        if (baseDataOffset > fileSize) {
            throw new IOException("Unexpected EOF while reading header");
        }

        ByteBuffer headerBuffer = ByteBuffer.allocate(headerLength);
        readFully(channel, headerBuffer, 8);
        headerBuffer.flip();

        JsonNode header = MAPPER.readTree(headerBuffer.array());
        return loadTensors(header, baseDataOffset, fileSize, (start, byteLength, elements, dtype) -> {
            ByteBuffer slice = channel.map(READ_ONLY, baseDataOffset + start, byteLength)
                .order(ByteOrder.LITTLE_ENDIAN);

            return readTensor(slice, elements, dtype);
        });
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
        int limit = buffer.limit();

        return loadTensors(header, baseDataOffset, limit, (start, byteLength, elements, dtype) -> {
            int pos = Math.toIntExact(baseDataOffset + start);
            int end = Math.toIntExact(baseDataOffset + start + byteLength);

            ByteBuffer slice = buffer.duplicate();
            slice.position(pos);
            slice.limit(end);
            slice = slice.slice().order(ByteOrder.LITTLE_ENDIAN);

            return readTensor(slice, elements, dtype);
        });
    }

    private static Map<String, Tensor> loadTensors(
        JsonNode header,
        long baseDataOffset,
        long totalSize,
        TensorReader reader
    ) {
        Map<String, Tensor> weights = new HashMap<>();

        header.fields().forEachRemaining(entry -> {
            String name = entry.getKey();

            if (name.equals("__metadata__")) return;

            JsonNode info = entry.getValue();
            int[] shape = parseShape(name, info.get("shape"));
            int elements = countElements(name, shape);
            TensorDType dtype = parseDType(name, info.get("dtype"));

            JsonNode offsets = info.has("offsets")
                ? info.get("offsets")
                : info.get("data_offsets");

            if (offsets == null || !offsets.isArray() || offsets.size() != 2) {
                throw Commons.illegalArgument("Invalid offsets for tensor: " + name);
            }

            long start = offsets.get(0).longValue();
            long end = offsets.get(1).longValue();

            if (start < 0 || end < start) {
                throw Commons.illegalArgument("Invalid offsets for tensor: " + name);
            }

            long actualLength = end - start;
            long tensorEnd = baseDataOffset + end;

            if (tensorEnd > totalSize) {
                throw Commons.illegalArgument("Tensor data exceeds safetensors payload: " + name);
            }

            long expectedLength = (long) elements * dtype.bytes();

            if (actualLength != expectedLength) {
                throw Commons.illegalArgument("Tensor size mismatch for tensor: %s! Expected %s but got %s bytes",
                    name, expectedLength, actualLength);
            }

            int byteLength = Math.toIntExact(actualLength);
            float[] values;

            try {
                values = reader.read(start, byteLength, elements, dtype);
            } catch (IOException e) {
                throw new RuntimeException("Failed to read tensor: " + name, e);
            }

            weights.put(name, Tensors.create(shape, values));
        });

        return weights;
    }

    private static int[] parseShape(String name, JsonNode shapeArray) {
        if (shapeArray == null || !shapeArray.isArray()) {
            throw Commons.illegalArgument("Invalid or missing shape for tensor: " + name);
        }

        int[] shape = new int[shapeArray.size()];

        for (int i = 0; i < shape.length; i++) {
            JsonNode dim = shapeArray.get(i);

            if (!dim.canConvertToInt()) {
                throw Commons.illegalArgument("Invalid shape dimension in tensor: " + name);
            }

            shape[i] = dim.intValue();
        }

        return shape;
    }

    private static int countElements(String name, int[] shape) {
        int elements = 1;

        for (int dim : shape) {
            if (dim < 0) {
                throw Commons.illegalArgument("Negative shape dimension in tensor: " + name);
            }

            elements = Math.multiplyExact(elements, dim);
        }

        return elements;
    }

    private static TensorDType parseDType(String name, JsonNode dtypeNode) {
        if (dtypeNode == null || !dtypeNode.isTextual()) {
            throw Commons.illegalArgument("Invalid or missing dtype for tensor: " + name);
        }

        return TensorDType.from(dtypeNode.textValue(), name);
    }

    private static float[] readTensor(ByteBuffer slice, int elements, TensorDType dtype) {
        return switch (dtype) {
            case F32 -> readF32Tensor(slice, elements);
            case F16 -> readF16Tensor(slice, elements);
        };
    }

    private static float[] readF32Tensor(ByteBuffer slice, int elements) {
        float[] values = new float[elements];
        slice.asFloatBuffer().get(values);
        return values;
    }

    private static float[] readF16Tensor(ByteBuffer slice, int elements) {
        float[] values = new float[elements];

        for (int i = 0; i < elements; i++) {
            values[i] = Commons.f16ToFloat(slice.getShort());
        }

        return values;
    }

    private static void readFully(FileChannel channel, ByteBuffer target, long position) throws IOException {
        while (target.hasRemaining()) {
            int read = channel.read(target, position + target.position());

            if (read < 0) {
                throw new IOException("Unexpected EOF while reading safetensors payload");
            }
        }
    }

    @FunctionalInterface
    private interface TensorReader {
        float[] read(long start, int byteLength, int elements, TensorDType dtype) throws IOException;
    }

    private enum TensorDType {
        F16(Short.BYTES),
        F32(Float.BYTES);

        private final int bytes;

        TensorDType(int bytes) {
            this.bytes = bytes;
        }

        public int bytes() {
            return bytes;
        }

        public static TensorDType from(String value, String tensorName) {
            return switch (value.toUpperCase()) {
                case "F16" -> F16;
                case "F32" -> F32;
                default -> throw Commons.illegalArgument("Unsupported dtype '%s' for tensor: %s", value, tensorName);
            };
        }
    }
}
