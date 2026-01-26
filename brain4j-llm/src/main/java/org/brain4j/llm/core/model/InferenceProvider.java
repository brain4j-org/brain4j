package org.brain4j.llm.core.model;

import org.brain4j.math.gpu.silicon.SiliconDevice;

import java.util.function.Consumer;

public interface InferenceProvider {
    String chat(String prompt);
    String chat(String prompt, SamplingConfig config);
    String chat(String prompt, SamplingConfig config, Consumer<String> tokenConsumer);
    LLM fork(SiliconDevice device);
}
