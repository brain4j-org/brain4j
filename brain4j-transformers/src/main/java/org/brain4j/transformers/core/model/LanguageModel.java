package org.brain4j.transformers.core.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.brain4j.core.importing.SafeTensors;
import org.brain4j.core.model.Model;
import org.brain4j.transformers.tokenizers.impl.BytePairTokenizer;
import org.brain4j.transformers.tokenizers.model.Tokenizer;
import org.brain4j.transformers.api.ModelFile;
import org.brain4j.transformers.api.ModelInfo;
import org.brain4j.transformers.core.architecture.ArchitectureAdapter;
import org.brain4j.transformers.core.architecture.ArchitectureRegistry;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.function.Consumer;

public class LanguageModel implements InferenceProvider {
    
    public static final ObjectMapper MAPPER = new ObjectMapper();
    
    private final String id;
    private final ModelInfo info;
    private final List<ModelFile> files;
    private final Map<String, Object> config;
    
    // Inference
    private Model model;
    private Tokenizer tokenizer;
    
    public LanguageModel(String id, ModelInfo info, List<ModelFile> files, Map<String, Object> config) {
        this.id = id;
        this.info = info;
        this.files = files;
        this.config = config;
    }
    
    public LanguageModel compile() throws IOException {
        this.tokenizer = new BytePairTokenizer();
        
        ModelFile configFile = findOrThrow("config.json", "config.json was not found!");
        ModelFile weightsFile = findOrThrow("model.safetensors", "model.safetensors was not found!");
        ModelFile tokenizerFile = findOrThrow("tokenizer.json", "tokenizer.json was not found!");
        
        JsonNode config = MAPPER.readTree(configFile.path().toFile());
        String modelType = config.get("model_type").asText();
        
        tokenizer.load(tokenizerFile.path().toFile());
        tokenizer.setBosTokenId(config.get("bos_token_id").asInt());
        tokenizer.setEosTokenId(config.get("eos_token_id").asInt());
        
        Map<String, Tensor> weights = SafeTensors.load(weightsFile.path());
        ArchitectureAdapter adapter = ArchitectureRegistry.findAdapter(modelType);
        this.model = adapter.buildModel(config, weights);

        return this;
    }
    
    @Override
    public String chat(String prompt) {
        return chat(prompt, SamplingConfig.defaultConfig());
    }
    
    @Override
    public String chat(String prompt, SamplingConfig config) {
        return chat(prompt, config, x -> {});
    }

    @Override
    public String chat(String prompt, SamplingConfig config, Consumer<String> tokenConsumer) {
        List<String> tokens = tokenizer.splitTokens(prompt);
        Tensor input = tokenizer.encode(tokens);

        StatesCache cache = new StatesCache();
        StringBuilder response = new StringBuilder(prompt);

        int bosToken = tokenizer.bosTokenId();
        int eosToken = tokenizer.eosTokenId();
        int generatedTokens = 0;

        if (bosToken != eosToken) input = input.concat(Tensors.scalar(bosToken));
        if (model.device() != null) input = input.to(model.device());

        Random random = config.random();
        Softmax activation = new Softmax(config.temperature());
        
        while (generatedTokens < config.maxLength()) {
            Tensor batchInput = input.unsqueeze(0);
            Tensor[] outs = model.predict(cache, batchInput);

            Tensor logits = outs[0].squeeze().cpu(); // [vocab_size]
            Tensor distribution = logits.activate(activation);
            
            float[] data = distribution.data();
            int[] topTokens = Tensors.topK(config.topK(), data);

            int chosen = random.nextInt(topTokens.length);
            int nextToken = topTokens[chosen];
            input = input.concat(Tensors.scalar(nextToken));

            String token = tokenizer.decode(nextToken);
            tokenConsumer.accept(token);

            if (nextToken == eosToken) break;
            
            response.append(token);
            generatedTokens++;
        }

        return response.toString();
    }

    @Override
    public LanguageModel fork(Device device) {
        if (model == null) throw new NullPointerException("Model has not been compiled!");
        if (device == null) throw new NullPointerException("Device cannot be null!");

        LanguageModel languageModel = new LanguageModel(id, info, files, config);
        languageModel.model = model.fork(device);
        return languageModel;
    }

    public Optional<ModelFile> find(String filename) {
        return files.stream().filter(file -> file.name().equals(filename)).findFirst();
    }
    
    private ModelFile findOrThrow(String filename, String message) throws FileNotFoundException {
        return find(filename).orElseThrow(() -> new FileNotFoundException(message));
    }
    
    public List<ModelFile> filesByFormat(String format) {
        return files.stream().filter(file -> file.format().equalsIgnoreCase(format)).toList();
    }
    
    public long totalSize() {
        return files.stream().mapToLong(ModelFile::size).sum();
    }
    
    public String getId() {
        return id;
    }
    
    public ModelInfo getInfo() {
        return info;
    }
    
    public List<ModelFile> getFiles() {
        return files;
    }
    
    public Map<String, Object> getConfig() {
        return config;
    }
    
    public Model getModel() {
        return model;
    }
    
    public Tokenizer getTokenizer() {
        return tokenizer;
    }
}
