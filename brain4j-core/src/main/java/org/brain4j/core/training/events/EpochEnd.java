package org.brain4j.core.training.events;

public record EpochEnd(int epoch, int totalEpochs) implements TrainingEvent {}
