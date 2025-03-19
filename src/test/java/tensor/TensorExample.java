package tensor;

import net.echo.brain4j.utils.math.tensor.Tensor;
import net.echo.brain4j.utils.math.tensor.TensorGPU;
import net.echo.brain4j.utils.math.tensor.TensorFactory;

public class TensorExample {

    /*
    * This test demonstrates Brain4J's GPU capabilities. 
    * It's not a complex neural network example or performance-optimized, 
    * as this is the first Brain4J release with full GPU support.
    */

    public static void main(String[] args) {
        System.out.println("XOR Neural Network Test with GPU and CPU:");
        
        System.out.println("GPU available: " + TensorGPU.isGpuAvailable());
        System.out.println("GPU usage: " + TensorFactory.isUsingGPU());
        
        TensorFactory.useGPUIfAvailable();
        
        long startCPU = System.currentTimeMillis();
        testXORNeuralNetworkCPU();
        long endCPU = System.currentTimeMillis();
        
        long startGPU = System.currentTimeMillis();
        testXORNeuralNetworkGPU();
        long endGPU = System.currentTimeMillis();
        
        System.out.println("\nPerformance Comparison:");
        System.out.println("CPU Time: " + (endCPU - startCPU) + " ms");
        System.out.println("GPU Time: " + (endGPU - startGPU) + " ms");
        System.out.println("Speedup: " + (double)(endCPU - startCPU) / (endGPU - startGPU) + "x");
        
        TensorGPU.releaseGPUResources();
    }
    
    private static void testXORNeuralNetworkCPU() {
        System.out.println("\nXOR Neural Network (CPU):");
        
        TensorFactory.forceCPU();
        
        xorNeuralNetwork();
    }
    
    private static void testXORNeuralNetworkGPU() {
        System.out.println("\nXOR Neural Network (GPU):");
        TensorFactory.useGPUIfAvailable();
        xorNeuralNetwork();
    }
    
    private static void xorNeuralNetwork() {
        int inputSize = 2;
        int hiddenSize = 3;
        int outputSize = 1;
        
        Tensor[] inputs = {
            TensorFactory.vector(0, 0),
            TensorFactory.vector(0, 1),
            TensorFactory.vector(1, 0),
            TensorFactory.vector(1, 1)
        };
        
        Tensor[] labels = {
            TensorFactory.vector(0),
            TensorFactory.vector(1),
            TensorFactory.vector(1),
            TensorFactory.vector(0)
        };
        
        Tensor W1 = TensorFactory.randn(0.0, 0.5, hiddenSize, inputSize);
        Tensor b1 = TensorFactory.zeros(hiddenSize);
        
        Tensor W2 = TensorFactory.randn(0.0, 0.5, outputSize, hiddenSize);
        Tensor b2 = TensorFactory.zeros(outputSize);
        
        double learningRate = 0.1;
        int epochs = 10000;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < inputs.length; i++) {
                Tensor hidden = W1.matmul(inputs[i].reshape(inputSize, 1));
                hidden.add(b1.reshape(hiddenSize, 1));
                
                Tensor hiddenActivated = hidden.clone().map(x -> Math.max(0, x));
                
                Tensor output = W2.matmul(hiddenActivated);
                output.add(b2.reshape(outputSize, 1));
                
                Tensor predicted = output.clone().map(x -> 1.0 / (1.0 + Math.exp(-x)));
                
                Tensor error = predicted.minus(labels[i].reshape(outputSize, 1));
                double loss = error.normSquared();
                totalLoss += loss;
                
                Tensor gradOutput = error.times(predicted.map(x -> x * (1 - x)));
                
                Tensor gradW2 = gradOutput.matmul(hiddenActivated.transpose());
                W2.sub(gradW2.times(learningRate));
                b2.sub(gradOutput.reshape(outputSize).times(learningRate));
                
                Tensor gradHidden = W2.transpose().matmul(gradOutput);
                
                Tensor gradHiddenActivated = gradHidden.clone();
                for (int j = 0; j < hiddenSize; j++) {
                    if (hidden.get(j, 0) <= 0) {
                        gradHiddenActivated.set(0, j, 0);
                    }
                }
                
                Tensor gradW1 = gradHiddenActivated.matmul(inputs[i].reshape(1, inputSize));
                W1.sub(gradW1.times(learningRate));
                b1.sub(gradHiddenActivated.reshape(hiddenSize).times(learningRate));
            }
            
            if ((epoch + 1) % 1000 == 0) {
                System.out.printf("Epoch %d: Average Loss = %.6f%n", epoch + 1, totalLoss / inputs.length);
            }
        }
        
        System.out.println("\nPredictions after training:");
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor hidden = W1.matmul(inputs[i].reshape(inputSize, 1));
            hidden.add(b1.reshape(hiddenSize, 1));
            
            Tensor hiddenActivated = hidden.clone().map(x -> Math.max(0, x));
            
            Tensor output = W2.matmul(hiddenActivated);
            output.add(b2.reshape(outputSize, 1));
            
            Tensor predicted = output.map(x -> 1.0 / (1.0 + Math.exp(-x)));
            
            System.out.printf("Input: [%.0f, %.0f], Output: %.6f, Expected: %.0f%n",
                    inputs[i].get(0), inputs[i].get(1), 
                    predicted.get(0, 0), labels[i].get(0));
        }
    }
}