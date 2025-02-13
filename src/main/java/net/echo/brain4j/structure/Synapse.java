package net.echo.brain4j.structure;

import net.echo.brain4j.structure.cache.Parameters;

import java.util.Random;

public class Synapse {

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private final int synapseId;
    private double weight;

    public Synapse(Random generator, Neuron inputNeuron, Neuron outputNeuron, double bound) {
        this.synapseId = Parameters.TOTAL_SYNAPSES++;
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
