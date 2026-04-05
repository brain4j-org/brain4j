package org.brain4j.core.importing.format.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.importing.LayerIO;
import org.brain4j.core.importing.SafeTensors;
import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.*;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static org.brain4j.core.importing.Registries.*;

public class BrainFormat implements BinaryFormat<Sequential> {

    public static final ObjectMapper MAPPER = new ObjectMapper();
    public static final int FORMAT_VERSION = 1;
    
    @Override
    public Sequential deserialize(File file) {
        Map<String, byte[]> files = readZip(file);
        return deserializeModel(files);
    }
    
    @Override
    public void serialize(Sequential model, File file) {
        if (model.device() != null) model = model.fork(null);
        
        Map<String, Tensor> weights = new HashMap<>();
        
        byte[] config = buildConfig(model, weights);
        byte[] weightData = buildWeights(weights);
        
        writeZip(file, Map.of(
            "config.json", config,
            "weights.safetensors", weightData
        ));
    }
    
    private Sequential deserializeModel(Map<String, byte[]> files) {
        try {
            byte[] specsData = files.get("config.json");
            byte[] weightsData = files.get("weights.safetensors");

            Map<String, Tensor> weights = SafeTensors.load(weightsData);
            JsonNode root = MAPPER.readTree(new String(specsData, StandardCharsets.UTF_8));

            int formatVersion = root.get("format_version").asInt();

            if (formatVersion != FORMAT_VERSION) {
                throw Commons.illegalArgument("Invalid format version: " + formatVersion);
            }

            JsonNode architecture = root.get("architecture");
            Map<Integer, Layer> architectureMap = new TreeMap<>();
            Map<Integer, List<String>> layerWeights = new TreeMap<>();

            for (JsonNode node : architecture) {
                int index = node.get("index").asInt();

                Layer layer = LayerIO.parse(node);
                architectureMap.put(index, layer);

                JsonNode connections = node.get("weights");
                List<String> ids = new ArrayList<>();

                for (JsonNode element : connections) {
                    if (!element.isTextual()) {
                        throw Commons.illegalArgument("All weights must be strings");
                    }
                    
                    ids.add(element.asText());
                }
                
                layerWeights.put(index, ids);
            }

            Layer[] layers = architectureMap.values().toArray(new Layer[0]);
            Sequential model = ModelSpecs.of(layers).compile();
            List<Layer> compiledLayers = model.getLayers();
            
            for (int i = 0; i < compiledLayers.size(); i++) {
                Layer layer = compiledLayers.get(i);
                Map<String, Tensor> params = layer.parameters();
                List<String> ids = layerWeights.getOrDefault(i, Collections.emptyList());
                
                for (String id : ids) {
                    String[] parts = id.split("\\.", 3);
                    if (parts.length != 3) {
                        throw Commons.illegalArgument("Invalid weight id format: %s", id);
                    }
                    
                    String name = parts[2];
                    Tensor param = weights.get(id);
                    
                    if (param == null) {
                        throw Commons.illegalArgument("Missing tensor '%s' in safetensors payload", id);
                    }
                    
                    params.put(name, param.withGrad());
                }
            }
            
            return model;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] buildConfig(Model model, Map<String, Tensor> globalWeights) {
        try {
            ObjectNode root = MAPPER.createObjectNode();
            root.put("format_version", FORMAT_VERSION);
            root.put("created_at", Instant.now().toString());

            ArrayNode architecture = MAPPER.createArrayNode();
            List<Layer> layers = model.getLayers();

            for (int i = 0; i < layers.size(); i++) {
                Layer layer = layers.get(i);
                String type = LAYER_REGISTRY.fromClass(layer.getClass());

                ObjectNode container = MAPPER.createObjectNode();
                ArrayNode weights = MAPPER.createArrayNode();

                LayerIO.write(layer, container);

                for (Map.Entry<String, Tensor> entry : layer.parameters().entrySet()) {
                    String fullName = entry.getKey();
                    String id = "%s.%s.%s".formatted(type, i, fullName);

                    globalWeights.put(id, entry.getValue());
                    weights.add(id);
                }

                container.put("index", i);
                container.put("type", type);
                container.set("weights", weights);

                architecture.add(container);
            }

            root.set("architecture", architecture);
            return MAPPER.writeValueAsBytes(root);
        } catch (Exception e) {
            throw new RuntimeException("Failed to build config", e);
        }
    }

    private byte[] buildWeights(Map<String, Tensor> weights) {
        try {
            ObjectNode header = MAPPER.createObjectNode();

            int offset = 0;
            Map<String, byte[]> rawData = new LinkedHashMap<>();

            for (var entry : weights.entrySet()) {
                String name = entry.getKey();
                Tensor tensor = entry.getValue();

                float[] values = tensor.data();
                int byteSize = values.length * 4;

                ObjectNode info = MAPPER.createObjectNode();

                ArrayNode shape = MAPPER.createArrayNode();
                for (int d : tensor.shape()) {
                    shape.add(d);
                }

                ArrayNode offsets = MAPPER.createArrayNode();
                offsets.add(offset);
                offsets.add(offset + byteSize);

                info.set("shape", shape);
                info.set("data_offsets", offsets);
                info.put("dtype", "F32");

                header.set(name, info);

                ByteBuffer buf = ByteBuffer
                    .allocate(byteSize)
                    .order(ByteOrder.LITTLE_ENDIAN);

                for (float v : values) {
                    buf.putFloat(v);
                }

                rawData.put(name, buf.array());
                offset += byteSize;
            }

            byte[] headerBytes = MAPPER.writeValueAsBytes(header);

            ByteBuffer result = ByteBuffer
                .allocate(8 + headerBytes.length + offset)
                .order(ByteOrder.LITTLE_ENDIAN);

            result.putLong(headerBytes.length);
            result.put(headerBytes);

            for (Map.Entry<String, byte[]> entry : rawData.entrySet()) {
                result.put(entry.getValue());
            }

            return result.array();

        } catch (Exception e) {
            throw new RuntimeException("Failed to build weights", e);
        }
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
