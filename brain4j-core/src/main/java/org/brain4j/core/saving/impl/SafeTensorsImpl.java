package org.brain4j.core.saving.impl;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.saving.ModelSerializer;
import org.brain4j.math.Commons;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class SafeTensorsImpl implements ModelSerializer {

    private record Metadata(List<String> activations, Map<String, Tensor> structure) {
    }

    public static Metadata parseStructure(File file) throws IOException {
        Map<String, Tensor> tensors = new HashMap<>();
        List<String> activations = new ArrayList<>();

        try (DataInputStream stream = new DataInputStream(new FileInputStream(file))) {
            ByteBuffer lenBuffer = ByteBuffer.wrap(stream.readNBytes(8));
            lenBuffer.order(ByteOrder.LITTLE_ENDIAN);

            int jsonLength = Math.toIntExact(lenBuffer.getLong());

            String json = new String(stream.readNBytes(jsonLength), StandardCharsets.UTF_8);
            JsonObject root = JsonParser.parseString(json).getAsJsonObject();

            for (Map.Entry<String, JsonElement> entry : root.entrySet()) {
                String key = entry.getKey();
                JsonObject obj = entry.getValue().getAsJsonObject();

                if (key.equals("__metadata__")) {
                    String[] activationsArray = new Gson().fromJson(obj.get("activations"), String[].class);
                    activations = Arrays.asList(activationsArray);
                }

                String dtype = obj.get("dtype").getAsString().toLowerCase();

                int[] shape = new Gson().fromJson(obj.get("shape"), int[].class);
                int[] dataOffsets = new Gson().fromJson(obj.get("data_offsets"), int[].class);

                int byteStart = dataOffsets[0];
                int byteEnd = dataOffsets[1];
                int byteLen = byteEnd - byteStart;

                byte[] data = stream.readNBytes(byteLen);
                ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

                Tensor tensor = Tensors.zeros(shape);
                float[] dest = tensor.data();

                for (int i = 0; i < dest.length; i++) {
                    float value = switch (dtype) {
                        case "f16" -> Commons.f16ToFloat(buffer.getShort());
                        case "f32", "f64" -> buffer.getFloat();
                        default -> throw new IllegalArgumentException("Unsupported dtype: " + dtype);
                    };
                    dest[i] = value;
                }

                tensors.put(key, tensor);
            }
        }

        return new Metadata(activations, tensors);
    }
    @Override
    public void serialize(File file, Model model) throws Exception {

    }

    @Override
    public Model deserialize(File file, Model model) throws Exception {
        Metadata metadata = parseStructure(file);
        Map<String, Tensor> tensors = metadata.structure();

        List<String> keys = tensors.keySet().stream().toList();
        List<Tensor> values = tensors.values().stream().toList();

        return model;
    }
}
