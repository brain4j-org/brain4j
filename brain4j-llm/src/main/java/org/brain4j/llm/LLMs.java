package org.brain4j.llm;

import org.brain4j.core.tokenizers.model.Tokenizer;
import org.brain4j.llm.core.loader.ModelLoader;
import org.brain4j.llm.core.loader.config.LoadConfig;
import org.brain4j.llm.core.model.LLM;

/**
 * Convenience static helpers for loading LLM models and tokenizers.
 *
 * <p>These methods provide a simple, high-level API for loading models
 * and tokenizers using the framework's default loader and configuration.
 * They wrap the lower-level {@link ModelLoader} usage and translate
 * exceptions into a consistent checked {@link Exception} type.
 *
 * <p>Example:
 * <pre>{@code
 * LLM model = LLMs.loadModel("gpt-small");
 * Tokenizer tok = LLMs.loadTokenizer("gpt-tokenizer");
 * }</pre>
 */
public class LLMs {

    /**
     * Load and compile a language model by its identifier.
     *
     * @param modelId the model id or path recognized by the loader
     * @return a compiled {@link LLM} instance ready for inference/training
     * @throws Exception if loading or compilation fails
     */
    public static LLM loadModel(String modelId) throws Exception {
        try (ModelLoader loader = new ModelLoader()) {
            return loader.loadModel(modelId, LoadConfig.defaultConfig()).compile();
        } catch (Exception e) {
            throw new Exception("Failed to load model: " + modelId, e);
        }
    }
    
    /**
     * Load a tokenizer by its identifier.
     *
     * @param tokenizerId the tokenizer id or path recognized by the loader
     * @return a {@link Tokenizer} instance
     * @throws Exception if the tokenizer cannot be loaded
     */
    public static Tokenizer loadTokenizer(String tokenizerId) throws Exception {
        try (ModelLoader loader = new ModelLoader()) {
            return loader.loadTokenizer(tokenizerId, LoadConfig.defaultConfig());
        } catch (Exception e) {
            throw new Exception("Failed to load tokenizer: " + tokenizerId, e);
        }
    }
}
