package org.brain4j.core.training.events;

public record EpochStart(int epoch, int totalEpochs) implements TrainingEvent {}
