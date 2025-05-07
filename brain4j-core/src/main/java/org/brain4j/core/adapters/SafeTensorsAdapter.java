package org.brain4j.core.adapters;

import com.google.gson.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class SafeTensorsAdapter {

    public static void load(String path) throws IOException {
        File file = new File(path);
        DataInputStream stream = new DataInputStream(new FileInputStream(file));

        byte[] lenBytes = stream.readNBytes(8);
        long jsonLength = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

        byte[] jsonBytes = stream.readNBytes((int) jsonLength);
        String json = new String(jsonBytes, StandardCharsets.UTF_8);

        System.out.println("Json size: " + jsonLength);
        System.out.println(json);

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonElement root = JsonParser.parseString(json);

        System.out.println(gson.toJson(root));

        JsonObject obj = root.getAsJsonObject();
        for (String key : obj.keySet()) {
            System.out.println("Key: " + key);

            if (key.equals("__metadata__")) continue;

            JsonObject tensorMeta = obj.getAsJsonObject(key);
            JsonArray shapeArray = tensorMeta.getAsJsonArray("shape");

            int[] shape = new int[shapeArray.size()];

            for (int i = 0; i < shapeArray.size(); i++) {
                shape[i] = shapeArray.get(i).getAsInt();
            }

            System.out.println("Shape: "+ Arrays.toString(shape));
        }
    }
}
