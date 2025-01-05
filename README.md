# Brain4J

[![Brain4J](https://img.shields.io/badge/Brain4J-2.5.1-blue.svg)](https://github.com/xEcho1337/Brain4J)

**Brain4J** is a powerful, lightweight, and easy-to-use Machine Learning library written in Java, designed for speed and simplicity.

Note: As of 2.2, Java 21 is required to make Brain4J work.

---

## Importing the library

As of 2.4, the library is now available to download on JitPack.

```
repositories {
    mavenCentral()
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.xEcho1337:brain4j:2.5.1'
}
```

## Getting Started

When building a neural network, you have many options. In this example, we will create a neural network to simulate an XOR gate.

### Defining the Model

To represent an XOR gate, we can use a simple neural network with four layers:

```java
Model network = new Model(
        new DenseLayer(2, Activations.LINEAR),
        new DenseLayer(16, Activations.RELU),
        new DenseLayer(16, Activations.RELU),
        new DenseLayer(1, Activations.SIGMOID)
);
```

### Compiling the Model

Next, define the weight initialization method and the loss function for training. Use the compile method as follows:

```java
network.compile(
        WeightInit.HE,
        LossFunctions.BINARY_CROSS_ENTROPY,
        new Adam(0.1),
        new StochasticUpdater()
);
```

For models with a single output neuron (producing values between 0 and 1), Binary Cross Entropy is the recommended loss function, paired with the Adam optimizer.

Also, when using the ReLU activation function it's suggested to use the `He` weight initialization for better results.

### Preparing Training Data

Create your training dataset using DataSet and DataRow:

```java
DataRow first = new DataRow(Vector.of(0, 0), Vector.of(0));
DataRow second = new DataRow(Vector.of(0, 1), Vector.of(1));
DataRow third = new DataRow(Vector.of(1, 0), Vector.of(1));
DataRow fourth = new DataRow(Vector.of(1, 1), Vector.of(0));

DataSet training = new DataSet(first, second, third, fourth);
training.partition(1);
```

### Training the Model

Once the setup is complete, use the fit method inside a loop to train the network. Training stops when the error is below a certain threshold.

	Tip: Always split your dataset into training and testing sets to evaluate the model’s performance.

```java
double error;

do {
    network.fit(training, 1);
    error = network.evaluate(training);
} while (error > 0.01);
```

The above code trains the neural network with a learning rate of 0.1, stopping when the error falls below 1%.

## Contributing & Contact

Contributions are always welcome via pull requests or issue reports.

- Telegram: @nettyfan
- Discord: @xecho1337
