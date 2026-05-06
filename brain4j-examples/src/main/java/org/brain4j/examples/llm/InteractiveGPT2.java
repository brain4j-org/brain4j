package org.brain4j.examples.llm;

import org.brain4j.llm.LLMs;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.llm.core.model.SamplingConfig;

import java.util.Scanner;

public class InteractiveGPT2 {
    public static void main(String[] args) throws Exception {
        new InteractiveGPT2().start();
    }

    public void start() throws Exception {
        LLM llm = LLMs.loadModel("Locutusque/gpt2-xl-conversational");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print(">>> ");
            String prompt = scanner.nextLine();

            if (prompt.equals("/exit")) break;

            SamplingConfig config = SamplingConfig.defaultConfig();

            llm.chat("<|USER|> " + prompt + " <|ASSISTANT|> ", config, System.out::print);
            System.out.println();
        }
    }
}
