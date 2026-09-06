package org.brain4j.examples.llm;

import org.brain4j.transformers.LLMs;
import org.brain4j.transformers.core.model.LanguageModel;
import org.brain4j.transformers.core.model.SamplingConfig;

import java.util.Scanner;

public class InteractiveGPT2 {
    public static void main(String[] args) throws Exception {
        new InteractiveGPT2().start();
    }

    public void start() throws Exception {
        LanguageModel languageModel = LLMs.loadModel("openai-community/gpt2-large");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print(">>> ");
            String prompt = scanner.nextLine();

            if (prompt.equals("/exit")) break;

            SamplingConfig config = SamplingConfig.builder()
                .maxLength(10)
                .setTemperature(0.0)
                .build();

            languageModel.chat(prompt, config, System.out::print);
            System.out.println();
        }
    }
}
