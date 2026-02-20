package org.brain4j.core.importing.format.impl;

import com.google.gson.*;
import org.brain4j.core.importing.SafeTensorsConverter;
import org.brain4j.core.importing.format.BinaryAdapter;
import org.brain4j.core.layer.Layer0;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.brain4j.core.importing.Registries.*;

public class BrainAdapter implements BinaryAdapter {
    
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final int FORMAT_VERSION = 2;
    
    @Override
    public Model deserialize(File file) {
        Map<String, byte[]> files = readZip(file);
        
        ModelSpecs specs = deserializeSpecs(files.get("metadata.json"));
        Model model = specs.compile(System.currentTimeMillis());
        
        deserializeWeights(model, files.get("weights.safetensors"));
        
        return model;
    }
    
    @Override
    public void serialize(Model model, File file) {
        if (model.getDevice() != null) model = model.fork(null);
        
        Map<String, Tensor> weights = new HashMap<>();
        
        byte[] metadata = buildMetadata(model, weights);
        byte[] weightData = buildWeights(weights);
        
        writeZip(file, Map.of(
            "metadata.json", metadata,
            "weights.safetensors", weightData
        ));
    }
    
    private ModelSpecs deserializeSpecs(byte[] data) {
        JsonObject root = GSON.fromJson(
            new String(data, StandardCharsets.UTF_8),
            JsonObject.class
        );
        
        JsonArray architecture = root.getAsJsonArray("architecture");
        ModelSpecs specs = ModelSpecs.of();
        
        for (JsonElement element : architecture) {
            JsonObject layerJson = element.getAsJsonObject();
            
            String type = layerJson.get("type").getAsString();
            String activation = layerJson.get("activation").getAsString();
            String clipper = layerJson.get("clipper").getAsString();
            
            Layer0 layer = LAYER_REGISTRY.toInstance(type);
            layer.setActivation(ACTIVATION_REGISTRY.toInstance(activation));
            layer.setClipper(CLIPPERS_REGISTRY.toInstance(clipper));
            
            layer.deserialize(layerJson);
            specs.add(layer);
        }
        
        return specs;
    }
    
    private void deserializeWeights(Model model, byte[] data) {
        Map<String, Tensor> tensors;
        
        try {
            tensors = SafeTensorsConverter.load(data);
        } catch (IOException e) {
            throw new RuntimeException("Failed to deserialize safe tensors!", e);
        }
        
        List<Layer0> layers = model.getLayers();
        Map<Layer0, Map<String, Tensor>> weightsPerLayer = new HashMap<>();
        
        for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
            String fullName = entry.getKey(); // es: dense.0.weights
            Tensor tensor = entry.getValue();
            
            String[] parts = fullName.split("\\.");
            if (parts.length < 3) {
                throw new IllegalStateException("Invalid weight name: " + fullName);
            }
            
            int layerIndex = Integer.parseInt(parts[1]);
            String paramName = parts[2];
            
            if (layerIndex < 0 || layerIndex >= layers.size()) {
                throw new IllegalStateException("Invalid layer index: " + layerIndex);
            }
            
            Layer0 layer = layers.get(layerIndex);
            
            weightsPerLayer
                .computeIfAbsent(layer, l -> new HashMap<>())
                .put(paramName, tensor);
        }
        
        for (var entry : weightsPerLayer.entrySet()) {
            Layer0 layer = entry.getKey();
            Map<String, Tensor> weights = entry.getValue();
            layer.loadWeights(weights);
        }
    }
    
    private byte[] buildMetadata(Model model, Map<String, Tensor> globalWeights) {
        JsonObject root = new JsonObject();
        root.addProperty("format_version", FORMAT_VERSION);
        root.addProperty("created_at", Instant.now().toString());
        
        JsonArray architecture = new JsonArray();
        List<Layer0> layers = model.getLayers();
        
        for (int i = 0; i < layers.size(); i++) {
            Layer0 layer = layers.get(i);
            JsonObject obj = new JsonObject();
            
            obj.addProperty("index", i);
            obj.addProperty("type", LAYER_REGISTRY.fromClass(layer.getClass()));
            obj.addProperty("activation", ACTIVATION_REGISTRY.fromClass(layer.getActivation().getClass()));
            obj.addProperty("clipper", CLIPPERS_REGISTRY.fromClass(layer.getClipper().getClass()));
            
            layer.serialize(obj);
            
            JsonArray weights = new JsonArray();
            for (var entry : layer.weightsMap().entrySet()) {
                String id = "%s.%d.%s".formatted(obj.get("type").getAsString(), i, entry.getKey());
                globalWeights.put(id, entry.getValue());
                weights.add(id);
            }
            
            obj.add("weights", weights);
            architecture.add(obj);
        }
        
        root.add("architecture", architecture);
        return GSON.toJson(root).getBytes(StandardCharsets.UTF_8);
    }
    
    private byte[] buildWeights(Map<String, Tensor> weights) {
        JsonObject header = new JsonObject();
        
        int offset = 0;
        Map<String, byte[]> rawData = new HashMap<>();
        
        for (var entry : weights.entrySet()) {
            String name = entry.getKey();
            Tensor tensor = entry.getValue();
            
            float[] values = tensor.data();
            int byteSize = values.length * 4;
            
            JsonObject info = new JsonObject();
            
            JsonArray shape = new JsonArray();
            for (int d : tensor.shape()) shape.add(d);
            
            JsonArray offsets = new JsonArray();
            offsets.add(offset);
            offsets.add(offset + byteSize);
            
            info.add("shape", shape);
            info.add("data_offsets", offsets);
            info.addProperty("dtype", "F32");
            
            header.add(name, info);
            
            ByteBuffer buf = ByteBuffer
                .allocate(byteSize)
                .order(ByteOrder.LITTLE_ENDIAN);
            
            for (float v : values) buf.putFloat(v);
            
            rawData.put(name, buf.array());
            offset += byteSize;
        }
        
        byte[] headerBytes = GSON.toJson(header).getBytes(StandardCharsets.UTF_8);
        
        ByteBuffer result = ByteBuffer
            .allocate(8 + headerBytes.length + offset)
            .order(ByteOrder.LITTLE_ENDIAN);
        
        result.putLong(headerBytes.length);
        result.put(headerBytes);
        
        for (byte[] data : rawData.values()) {
            result.put(data);
        }
        
        return result.array();
    }
            
    private Map<String, byte[]> readZip(File file) {
        Map<String, byte[]> result = new HashMap<>();
        
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(file))) {
            ZipEntry entry;
            byte[] buffer = new byte[8192];
            
            while ((entry = zis.getNextEntry()) != null) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                int read;
                while ((read = zis.read(buffer)) != -1) {
                    baos.write(buffer, 0, read);
                }
                result.put(entry.getName(), baos.toByteArray());
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read model zip", e);
        }
        
        return result;
    }
    
    private void writeZip(File file, Map<String, byte[]> files) {
        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(file))) {
            zos.setLevel(Deflater.BEST_SPEED);
            
            for (var entry : files.entrySet()) {
                ZipEntry ze = new ZipEntry(entry.getKey());
                zos.putNextEntry(ze);
                zos.write(entry.getValue());
                zos.closeEntry();
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write model zip", e);
        }
    }
}
