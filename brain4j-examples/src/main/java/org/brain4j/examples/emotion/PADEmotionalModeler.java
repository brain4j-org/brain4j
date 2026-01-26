package org.brain4j.examples.emotion;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.loss.impl.MeanSquaredError;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.DefaultTrainer;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.examples.emotion.registry.EmotionRegistry;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.ArrayList;
import java.util.List;

public class PADEmotionalModeler {

    public static List<Tensor> createFeatureData() {
        List<Tensor> features = new ArrayList<>();
        features.add(Tensors.vector(0.5f, 0.3f, 0.8f)); // input for "Joy"
        features.add(Tensors.vector(0.2f, 0.7f, 0.1f)); // input for "Anger"
        features.add(Tensors.vector(0.1f, 0.0f, 0.0f)); // input for "Sadness"
        return features;
    }

    public static List<Tensor> createPADLabels() {
        List<Tensor> labels = new ArrayList<>();
        labels.add(Tensors.vector(
            (float) EmotionRegistry.getEmotion("Joy").getPleasure(),
            (float) EmotionRegistry.getEmotion("Joy").getArousal(),
            (float) EmotionRegistry.getEmotion("Joy").getDominance()
        ));
        labels.add(Tensors.vector(
            (float) EmotionRegistry.getEmotion("Anger").getPleasure(),
            (float) EmotionRegistry.getEmotion("Anger").getArousal(),
            (float) EmotionRegistry.getEmotion("Anger").getDominance()
        ));
        labels.add(Tensors.vector(
            (float) EmotionRegistry.getEmotion("Sadness").getPleasure(),
            (float) EmotionRegistry.getEmotion("Sadness").getArousal(),
            (float) EmotionRegistry.getEmotion("Sadness").getDominance()
        ));
        return labels;
    }

    public static void main(String[] args) {
        List<Tensor> trainingFeatures = createFeatureData();
        List<Tensor> trainingLabels = createPADLabels();

        if (trainingFeatures.isEmpty() || trainingLabels.isEmpty() || trainingFeatures.size() != trainingLabels.size()) {
            System.err.println("Training data is invalid or empty. Cannot proceed with model training.");
            return;
        }

        int inputSize = trainingFeatures.getFirst().elements();
        int outputSize = 3; // for P, A, D

        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(inputSize),
            new DenseLayer(64, Activations.RELU).setWeightInit(new UniformXavierInit()),
            new DenseLayer(32, Activations.RELU).setWeightInit(new UniformXavierInit()),
            new DenseLayer(outputSize, Activations.TANH).setWeightInit(new UniformXavierInit())
        );

        Model model = specs.compile();
        TrainingConfig config = TrainingConfig.of(new MeanSquaredError(), new Adam(0.001));
        
        int epochs = 100;
        int batchSize = 1;

        List<Sample> samples = new ArrayList<>();
        for (int i = 0; i < trainingFeatures.size(); i++) {
            samples.add(new Sample(trainingFeatures.get(i), trainingLabels.get(i)));
        }
        ListDataSource trainSource = new ListDataSource(samples, false, batchSize);
        Trainer trainer = Trainer.of(model, config);
        System.out.println("Starting conceptual training...");
        for (int i = 0; i < epochs; i++) {
            trainer.fit(trainSource.clone(), 1);
            double avgLoss = model.evaluate(trainSource.clone(), config.loss()).loss();
            System.out.printf("Epoch %d, Average Loss: %.4f%n", (i + 1), avgLoss);
        }
        System.out.println("Conceptual training finished.");

        System.out.println("\nMaking predictions:");
        Tensor newFeatures = Tensors.vector(0.4f, 0.4f, 0.7f);
        Tensor predictedPAD = model.predict(newFeatures).flatten();
        System.out.println("Input features: " + newFeatures);
        System.out.println("Predicted PAD: " + predictedPAD);

        PADEmotionalState predictedState = new PADEmotionalState(
                predictedPAD.get(0), predictedPAD.get(1), predictedPAD.get(2)
        );
        System.out.println("Predicted Emotional State: " + predictedState);

        String closestEmotion = null;
        double minDistance = Double.MAX_VALUE;
        for (var entry : EmotionRegistry.EMOTIONS.entrySet()) {
            double distance = predictedState.euclideanDistance(entry.getValue());
            if (distance < minDistance) {
                minDistance = distance;
                closestEmotion = entry.getKey();
            }
        }
        System.out.println("Closest predefined emotion: " + closestEmotion + " (Distance: " + String.format("%.2f", minDistance) + ")");
    }
}
