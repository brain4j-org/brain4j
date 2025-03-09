package conv;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.layer.impl.convolution.FlattenLayer;
import net.echo.brain4j.layer.impl.convolution.InputLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class ConvExample {

    public static void main(String[] args) throws IOException {
        ConvExample example = new ConvExample();
        example.start();
    }

    private void start() throws IOException {
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        System.out.println(model.getStats());

        // model.fit(dataSet, 10);
        // model.save("mnist-conv.json");

        EvaluationResult result = model.evaluate(dataSet);

        System.out.println(result.confusionMatrix());
    }

    private Sequential getModel() {
        Sequential model = new Sequential(
                // Input layer, necessary when using CNNs
                new InputLayer(28, 28),

                // #1 convolutional block
                new ConvLayer(32, 3, 3, Activations.MISH),

                // #2 convolutional block
                new ConvLayer(32, 5, 5, Activations.MISH),

                // Flattens the feature map to a 1D vector
                new FlattenLayer(484), // You must find the right size by trial and error

                // Classifiers
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.005));
    }

    private DataSet<DataRow> getDataSet() throws IOException {
        DataSet<DataRow> dataSet = new DataSet<>();

        FileReader reader = new FileReader("dataset.csv");
        CSVParser parser = new CSVParser(reader, CSVFormat.EXCEL);

        parser.forEach(record -> {
            List<String> columns = record.toList();

            String label = columns.getFirst();
            List<String> pixels = columns.subList(1, columns.size());

            Vector output = new Vector(10);
            output.set(Integer.parseInt(label), 1);

            Vector input = Vector.parse(pixels).divide(255);
            dataSet.add(new DataRow(input, output));
        });

        return dataSet;
    }
}
