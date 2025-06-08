package org.brain4j.core.training.wrappers;

import org.brain4j.math.data.ListDataSource;

public class TrainingParams {

    private ListDataSource trainSource;
    private ListDataSource validationSource;
    private int epochs;
    private int evaluateEvery;

    public TrainingParams(ListDataSource trainSource) {
        this.trainSource = trainSource;
        this.validationSource = trainSource;
    }

    public TrainingParams(ListDataSource trainSource, ListDataSource validationSource) {
        this.trainSource = trainSource;
        this.validationSource = validationSource;
    }

    public ListDataSource train() {
        return trainSource;
    }

    public TrainingParams train(ListDataSource trainSource) {
        this.trainSource = trainSource;
        return this;
    }

    public ListDataSource validation() {
        return validationSource;
    }

    public TrainingParams validation(ListDataSource validationSource) {
        this.validationSource = validationSource;
        return this;
    }

    public int epochs() {
        return epochs;
    }

    public TrainingParams epochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public int evaluateEvery() {
        return evaluateEvery;
    }

    public TrainingParams evaluateEvery(int evaluateEvery) {
        this.evaluateEvery = evaluateEvery;
        return this;
    }
}
