package org.brain4j.transformers.core.model;

import org.brain4j.math.gpu.device.Device;

import java.util.function.Consumer;

public interface InferenceProvider {
    String chat(String prompt);
    String chat(String prompt, SamplingConfig config);
    String chat(String prompt, SamplingConfig config, Consumer<String> tokenConsumer);
    LanguageModel fork(Device device);
}
