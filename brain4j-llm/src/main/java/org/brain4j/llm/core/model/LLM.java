package org.brain4j.llm.core.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import org.brain4j.core.importing.SafeTensors;
import org.brain4j.core.model.Model;
import org.brain4j.core.transformer.tokenizers.impl.BytePairTokenizer;
import org.brain4j.core.transformer.tokenizers.model.Tokenizer;
import org.brain4j.llm.api.ModelFile;
import org.brain4j.llm.api.ModelInfo;
import org.brain4j.llm.core.architecture.ArchitectureAdapter;
import org.brain4j.llm.core.architecture.ArchitectureRegistry;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.function.Consumer;

public class LLM implements InferenceProvider {
    
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    
    private final String id;
    private final ModelInfo info;
    private final List<ModelFile> files;
    private final Map<String, Object> config;
    
    // Inference
    private Model model;
    private Tokenizer tokenizer;
    
    public LLM(String id, ModelInfo info, List<ModelFile> files, Map<String, Object> config) {
        this.id = id;
        this.info = info;
        this.files = files;
        this.config = config;
    }
    
    public LLM compile() throws IOException {
        this.tokenizer = new BytePairTokenizer();
        
        ModelFile configFile = findOrThrow("config.json", "config.json was not found!");
        ModelFile weightsFile = findOrThrow("model.safetensors", "model.safetensors was not found!");
        ModelFile tokenizerFile = findOrThrow("tokenizer.json", "tokenizer.json was not found!");
        
        String configText = new String(Files.readAllBytes(configFile.path()));
        JsonObject config = GSON.fromJson(configText, JsonObject.class);
        String modelType = config.get("model_type").getAsString();
        
        tokenizer.load(tokenizerFile.path().toFile());
        tokenizer.setBosTokenId(config.get("bos_token_id").getAsInt());
        tokenizer.setEosTokenId(config.get("eos_token_id").getAsInt());
        
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
    public LLM fork(Device device) {
        if (model == null) throw new NullPointerException("Model has not been compiled!");
        if (device == null) throw new NullPointerException("Device cannot be null!");

        LLM llm = new LLM(id, info, files, config);
        llm.model = model.fork(device);
        return llm;
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
