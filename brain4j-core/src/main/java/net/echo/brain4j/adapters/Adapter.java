package net.echo.brain4j.adapters;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public interface Adapter {

    void serialize(DataOutputStream stream) throws IOException;

    void deserialize(DataInputStream stream) throws IOException;
}
