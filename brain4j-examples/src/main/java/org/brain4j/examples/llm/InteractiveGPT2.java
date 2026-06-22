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
        LanguageModel languageModel = LLMs.loadModel("Locutusque/gpt2-xl-conversational");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print(">>> ");
            String prompt = scanner.nextLine();

            if (prompt.equals("/exit")) break;

            SamplingConfig config = SamplingConfig.defaultConfig();

            languageModel.chat("<|USER|> " + prompt + " <|ASSISTANT|> ", config, System.out::print);
            System.out.println();
        }
    }
}
