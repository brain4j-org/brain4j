package org.brain4j.core.serializing.impl;

import org.brain4j.core.serializing.ModelAdapter;
import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Brain4JUtils;

import java.io.*;

public class BrainFormatAdapter implements ModelAdapter {

    @Override
    public void serialize(String path, Model model) throws Exception {
        String suffix = path.endsWith(".b4j") ? "" : ".b4j";
        serialize(new File(path + suffix), model);
    }

    @Override
    public void serialize(File file, Model model) throws Exception {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        DataOutputStream dataStream = new DataOutputStream(outputStream);

        dataStream.writeUTF("2.9.0");
        dataStream.writeInt(model.getSeed()); // seed
        dataStream.writeUTF(model.getLossFunction().getClass().getName());
        dataStream.writeUTF(model.getWeightInit().getClass().getName());
        dataStream.writeUTF(model.getUpdater().getClass().getName());
        dataStream.writeUTF(model.getOptimizer().getClass().getName());

        model.getOptimizer().serialize(dataStream); // optimizer
        model.serialize(dataStream); // serializes layers

        byte[] bytes = outputStream.toByteArray();

        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            byte[] compressed = ModelAdapter.compress(bytes);
            fileOutputStream.write(compressed);
        }
    }

    @Override
    public Model deserialize(String path, Model model) throws Exception {
        return deserialize(new File(path), model);
    }

    @Override
    public Model deserialize(File file, Model model) throws Exception {
        try (FileInputStream fileInputStream = new FileInputStream(file)) {
            byte[] bytes = ModelAdapter.decompress(fileInputStream.readAllBytes());
            DataInputStream dataStream = new DataInputStream(new ByteArrayInputStream(bytes));

            String version = dataStream.readUTF();
            int seed = dataStream.readInt();
            LossFunction lossFunction = Brain4JUtils.newInstance(dataStream.readUTF());
            WeightInitializer weightInit = Brain4JUtils.newInstance(dataStream.readUTF());
            Updater updater = Brain4JUtils.newInstance(dataStream.readUTF());
            Optimizer optimizer = Brain4JUtils.newInstance(dataStream.readUTF());

            optimizer.deserialize(dataStream);

            model.setSeed(seed);
            model.deserialize(dataStream);

            model.setLossFunction(lossFunction);
            model.setWeightInit(weightInit);
            model.setUpdater(updater);
            model.setOptimizer(optimizer);

            model.getUpdater().resetGradients();
            model.getOptimizer().postInitialize(model);
            model.setPropagation(new BackPropagation(model, optimizer, updater));

            for (Layer layer : model.getLayers()) {
                layer.compile(weightInit, lossFunction, optimizer, updater);
            }

            return model;
        }
    }
}
