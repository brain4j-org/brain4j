package org.brain4j.transformers.api.huggingface;

import org.apache.hc.core5.http.ClassicHttpResponse;

import java.io.IOException;
import java.io.InputStream;

public record FileDownloadResponse(ClassicHttpResponse response, InputStream inputStream) implements AutoCloseable {

    @Override
    public void close() throws IOException {
        try {
            if (inputStream != null) {
                inputStream.close();
            }
        } finally {
            if (response != null) {
                response.close();
            }
        }
    }
}