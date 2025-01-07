package net.echo.brain4j.structure;

import java.util.Random;

public class Synapse {

    public static int SYNAPSE_COUNTER = 0;

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private final int synapseId;
    private double weight;

    public Synapse(Random generator, Neuron inputNeuron, Neuron outputNeuron, double bound) {
        this.synapseId = SYNAPSE_COUNTER++;
        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;
        this.weight = (generator.nextDouble() * 2 * bound) - bound;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public int getSynapseId() {
        return synapseId;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
