package org.brain4j.datasets.api;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ClassicHttpResponse;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.brain4j.datasets.api.exception.DatasetException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

public class HuggingFaceClient implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceClient.class);
    private static final String BASE_URL = "https://huggingface.co";
    private static final String API_BASE_URL = "https://huggingface.co/api";

    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String userAgent;

    public HuggingFaceClient() {
        this.httpClient = HttpClients.createDefault();
        this.objectMapper = new ObjectMapper()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        this.userAgent = "brain4j-datasets/1.0.0";
    }

    public Optional<DatasetInfo> getDatasetInfo(String datasetId) throws DatasetException {
        validateDatasetId(datasetId);

        String encodedId = URLEncoder.encode(datasetId, StandardCharsets.UTF_8);
        String url = API_BASE_URL + "/datasets/" + encodedId;

        try {
            HttpGet request = new HttpGet(URI.create(url));
            request.setHeader("User-Agent", userAgent);

            try (CloseableHttpResponse response = httpClient.execute(request)) {
                return processDatasetInfoResponse(response, datasetId);
            }
        } catch (IOException e) {
            throw new DatasetException("Network error while retrieving dataset info for: " + datasetId, e);
        } catch (ParseException e) {
            throw new DatasetException("Failed to parse dataset info response for: " + datasetId, e);
        }
    }

    public FileDownloadResponse downloadFile(String datasetId, String filename) throws DatasetException {
        validateDatasetId(datasetId);
        validateFilename(filename);

        String encodedId = URLEncoder.encode(datasetId, StandardCharsets.UTF_8);
        String encodedFilename = URLEncoder.encode(filename, StandardCharsets.UTF_8);
        String url = BASE_URL + "/datasets/" + encodedId + "/resolve/main/" + encodedFilename;

        try {
            HttpGet request = new HttpGet(URI.create(url));
            request.setHeader("User-Agent", userAgent);

            CloseableHttpResponse response = httpClient.execute(request);

            if (response.getCode() != 200) {
                String errorMessage = "Failed to download file. Status: " + response.getCode();
                try {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    errorMessage += ", Body: " + responseBody;
                } catch (IOException ignored) {
                    // ignore parsing errors for error response
                } finally {
                    try {
                        response.close();
                    } catch (IOException ignored) {
                        // ignore close errors
                    }
                }
                throw new DatasetException(errorMessage);
            }

            return new FileDownloadResponse(response, response.getEntity().getContent());
        } catch (IOException e) {
            throw new DatasetException("Network error while downloading file: " + filename, e);
        } catch (ParseException e) {
            throw new DatasetException("Failed to parse file download response for: " + filename, e);
        }
    }

    private Optional<DatasetInfo> processDatasetInfoResponse(ClassicHttpResponse response, String datasetId)
            throws DatasetException, IOException, ParseException {
        int statusCode = response.getCode();
        String responseBody = EntityUtils.toString(response.getEntity());

        logger.debug("Response status: {}, body length: {}", statusCode, responseBody.length());
        if (logger.isTraceEnabled()) {
            logger.trace("Response body: {}", responseBody);
        }

        return switch (statusCode) {
            case 200 -> {
                try {
                    DatasetInfo info = objectMapper.readValue(responseBody, DatasetInfo.class);
                    logger.debug("Successfully retrieved dataset info for: {}", datasetId);
                    yield Optional.of(info);
                } catch (Exception e) {
                    logger.error("Failed to parse JSON response for dataset: {}", datasetId, e);
                    logger.error("Response body that failed to parse: {}", responseBody);
                    throw new DatasetException("Failed to parse dataset info response: " + e.getMessage(), e);
                }
            }
            case 404 -> {
                logger.warn("Dataset not found: {}", datasetId);
                yield Optional.empty();
            }
            default -> throw new DatasetException(
                    "Failed to retrieve dataset info. Status: " + statusCode + ", Body: " + responseBody
            );
        };
    }

    private void validateDatasetId(String datasetId) {
        if (datasetId == null || datasetId.trim().isEmpty()) {
            throw new IllegalArgumentException("Dataset ID cannot be null or empty");
        }
    }

    private void validateFilename(String filename) {
        if (filename == null || filename.trim().isEmpty()) {
            throw new IllegalArgumentException("Filename cannot be null or empty");
        }
    }

    @Override
    public void close() throws IOException {
        if (httpClient != null) {
            httpClient.close();
        }
    }
}