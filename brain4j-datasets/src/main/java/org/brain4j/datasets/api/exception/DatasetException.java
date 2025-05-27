package org.brain4j.datasets.api.exception;

public class DatasetException extends Exception {

    public DatasetException(String message) {
        super(message);
    }

    public DatasetException(String message, Throwable cause) {
        super(message, cause);
    }
}