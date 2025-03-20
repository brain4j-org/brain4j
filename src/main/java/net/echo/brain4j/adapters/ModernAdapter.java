package net.echo.brain4j.adapters;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.MLUtils;

import java.io.*;
import java.util.List;

public class ModernAdapter {

    public static void serialize(String path, Model<?, ?, ?> model) throws Exception {
        serialize(new File(path), model);
    }

    public static void serialize(File file, Model<?, ?, ?> model) throws Exception {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        DataOutputStream dataStream = new DataOutputStream(outputStream);

        dataStream.writeInt(model.getSeed()); // seed
        dataStream.writeUTF(model.getLossFunction().getClass().getName()); // loss_function
        dataStream.writeUTF(model.getWeightInit().getClass().getName()); // weight_init
        dataStream.writeUTF(model.getUpdater().getClass().getName()); // updater
        dataStream.writeUTF(model.getOptimizer().getClass().getName());

        model.getOptimizer().serialize(dataStream); // optimizer
        model.serialize(dataStream); // serializes layers

        int layers = model.getLayers().size(); // layers

        for (int i = 0; i < layers; i++) {
            Layer<?, ?> layer = model.getLayers().get(i);

            for (Neuron neuron : layer.getNeurons()) {
                dataStream.writeDouble(neuron.getBias());
            }

            int synapses = layer.getSynapses().size();

            for (int j = 0; j < synapses; j++) {
                Synapse synapse = layer.getSynapses().get(j);
                dataStream.writeDouble(synapse.getWeight());
            }
        }

        byte[] bytes = outputStream.toByteArray();

        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            fileOutputStream.write(bytes);
        }
    }

    public static <T extends Model<?, ?, ?>> T deserialize(String path, T model) throws Exception {
        return deserialize(new File(path), model);
    }

    public static <T extends Model<?, ?, ?>> T deserialize(File file, T model) throws Exception {
        try (FileInputStream fileInputStream = new FileInputStream(file)) {
            DataInputStream dataStream = new DataInputStream(fileInputStream);
            int seed = dataStream.readInt();

            String lossFunctionClass = dataStream.readUTF();
            String weightInitClass = dataStream.readUTF();
            String updaterClass = dataStream.readUTF();
            String optimizerClass = dataStream.readUTF();

            LossFunction lossFunction = MLUtils.newInstance(lossFunctionClass);
            WeightInitializer weightInit = MLUtils.newInstance(weightInitClass);
            Updater updater = MLUtils.newInstance(updaterClass);
            Optimizer optimizer = MLUtils.newInstance(optimizerClass);

            optimizer.deserialize(dataStream);

            model.setSeed(seed);
            model.deserialize(dataStream);

            model.compile(weightInit, lossFunction, optimizer, updater);

            int layers = model.getLayers().size();

            for (int i = 0; i < layers; i++) {
                Layer<?, ?> layer = model.getLayers().get(i);

                List<Neuron> neurons = layer.getNeurons();

                for (Neuron neuron : neurons) {
                    double bias = dataStream.readDouble();
                    neuron.setBias(bias);
                }

                List<Synapse> synapses = layer.getSynapses();

                for (Synapse synapse : synapses) {
                    double weight = dataStream.readDouble();
                    synapse.setWeight(weight);
                }
            }

            model.reloadWeights();
            return model;
        }
    }
}
