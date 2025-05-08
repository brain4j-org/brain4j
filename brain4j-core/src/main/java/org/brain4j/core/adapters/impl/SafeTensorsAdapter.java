package org.brain4j.core.adapters.impl;

import com.google.gson.*;
import org.brain4j.core.adapters.ModelAdapter;
import org.brain4j.core.model.Model;
import org.brain4j.math.BrainUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class SafeTensorsAdapter implements ModelAdapter {

    public static Model load(String path, Model model) throws IOException {
        try (DataInputStream stream = new DataInputStream(new FileInputStream(path))) {
            long jsonLength = ByteBuffer.wrap(stream.readNBytes(8))
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .getLong();

            String json = new String(stream.readNBytes((int) jsonLength), StandardCharsets.UTF_8);
            JsonObject root = JsonParser.parseString(json).getAsJsonObject();

            Map<String, Tensor> tensors = new HashMap<>();

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

            for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
                System.out.println(entry.getKey());
            }
        }

        return model;
    }

    @Override
    public void serialize(String path, Model model) throws Exception {

    }

    @Override
    public void serialize(File file, Model model) throws Exception {

    }

    @Override
    public Model deserialize(String path, Model model) throws Exception {
        return null;
    }

    @Override
    public Model deserialize(File file, Model model) throws Exception {
        return null;
    }
}
