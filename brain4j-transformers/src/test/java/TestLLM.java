import org.brain4j.transformers.LLMs;
import org.brain4j.transformers.core.model.LanguageModel;
import org.brain4j.transformers.core.model.SamplingConfig;

import java.util.function.Consumer;

public class TestLLM {

    public static void main(String[] args) throws Exception {
        LanguageModel languageModel = LLMs.loadModel("gpt2");

        SamplingConfig config = SamplingConfig.builder().maxLength(256).build();
        TokenHandler handler = new TokenHandler();
        String prompt = "Hello, my name is";

        languageModel.getModel().summary();
        languageModel.chat(prompt, config, handler);
        handler.printStats();
    }
    
    private static class TokenHandler implements Consumer<String> {
        
        private long lastTokenTime;
        private double totalTime;
        private int generatedTokens;
        
        public TokenHandler() {
            this.lastTokenTime = System.nanoTime();
        }
        
        @Override
        public void accept(String s) {
            long now = System.nanoTime();
            double took = (now - lastTokenTime) / 1e6;
            System.out.print(s);
            
            this.lastTokenTime = now;
            this.totalTime += took;
            this.generatedTokens++;
        }
        
        public void printStats() {
            double average = totalTime / generatedTokens;
            
            System.out.println();
            System.out.printf("%s generated tokens %n", generatedTokens);
            System.out.printf("total ms  = %.2f %n", totalTime);
            System.out.printf("avg/token = %.2f %n", average);
        }
    }
}
