package org.brain4j.core.adapters.impl;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.brain4j.core.adapters.ModelAdapter;
import org.brain4j.core.model.Model;
import org.brain4j.math.BrainUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class SafeTensorsAdapter implements ModelAdapter {

    public static Map<String, Tensor> parseStructure(File file) throws IOException {
        Map<String, Tensor> tensors = new HashMap<>();

        try (DataInputStream stream = new DataInputStream(new FileInputStream(file))) {
            ByteBuffer lenBuffer = ByteBuffer.wrap(stream.readNBytes(8));
            lenBuffer.order(ByteOrder.LITTLE_ENDIAN);

            int jsonLength = Math.toIntExact(lenBuffer.getLong());

            String json = new String(stream.readNBytes(jsonLength), StandardCharsets.UTF_8);
            JsonObject root = JsonParser.parseString(json).getAsJsonObject();

            for (Map.Entry<String, JsonElement> entry : root.entrySet()) {
                String key = entry.getKey();
                if (key.equals("__metadata__")) continue;

                JsonObject obj = entry.getValue().getAsJsonObject();
                String dtype = obj.get("dtype").getAsString().toLowerCase();

                int[] shape = new Gson().fromJson(obj.get("shape"), int[].class);
                int[] dataOffsets = new Gson().fromJson(obj.get("data_offsets"), int[].class);

                int byteStart = dataOffsets[0];
                int byteEnd = dataOffsets[1];
                int byteLen = byteEnd - byteStart;

                byte[] data = stream.readNBytes(byteLen);
                ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

                Tensor tensor = Tensors.create(shape);
                float[] dest = tensor.getData();

                switch (dtype) {
                    case "f16" -> {
                        for (int i = 0; i < dest.length; i++) {
                            short f16 = buffer.getShort();
                            dest[i] = BrainUtils.f16ToFloat(f16);
                        }
                    }
                    case "f32", "f64" -> {
                        for (int i = 0; i < dest.length; i++) {
                            dest[i] = buffer.getFloat();
                        }
                    }
                }

                tensors.put(key, tensor);
            }
        }

        return tensors;
    }

    @Override
    public void serialize(String path, Model model) throws Exception {

    }

    @Override
    public void serialize(File file, Model model) throws Exception {

    }

    @Override
    public Model deserialize(String path, Model model) throws Exception {
        return deserialize(new File(path), model);
    }

    @Override
    public Model deserialize(File file, Model model) throws Exception {
        Map<String, Tensor> structure = parseStructure(file);

        structure.forEach((key, tensor) -> {
            String[] paths = key.split("\\.");

        });

        return model;
    }
}
