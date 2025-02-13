package mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

public class DigitRecognizerApp extends JFrame {

    private final Model model;
    private DrawingPanel drawingPanel;
    private PredictionPanel predictionPanel;
    private JToggleButton penButton;

    public DigitRecognizerApp(Model model) {
        this.model = model;
        initComponents();
    }

    private void initComponents() {
        setTitle("Digit Recognizer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        JPanel toolPanel = new JPanel();
        penButton = new JToggleButton("Penna");
        JToggleButton eraserButton = new JToggleButton("Gomma");
        ButtonGroup toolGroup = new ButtonGroup();
        toolGroup.add(penButton);
        toolGroup.add(eraserButton);
        penButton.setSelected(true);
        toolPanel.add(penButton);
        toolPanel.add(eraserButton);

        JButton clearButton = new JButton("Pulisci");
        toolPanel.add(clearButton);
        clearButton.addActionListener(e -> {
            drawingPanel.clear();
            updatePrediction();
        });

        drawingPanel = new DrawingPanel();
        drawingPanel.setPreferredSize(new Dimension(28 * drawingPanel.getScale(), 28 * drawingPanel.getScale()));
        drawingPanel.setDrawingListener(this::updatePrediction);

        predictionPanel = new PredictionPanel();
        predictionPanel.setPreferredSize(new Dimension(200, 28 * drawingPanel.getScale()));

        add(toolPanel, BorderLayout.NORTH);
        add(drawingPanel, BorderLayout.CENTER);
        add(predictionPanel, BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void updatePrediction() {
        double[] imageData = drawingPanel.getImageData();
        Vector input = new Vector(784);

        for (int i = 0; i < 784; i++) {
            input.set(i, imageData[i]);
        }

        Vector output = model.predict(input);
        int predictedDigit = MLUtils.indexOfMaxValue(output);
        predictionPanel.updatePrediction(predictedDigit, output);
    }

    public boolean isPenSelected() {
        return penButton.isSelected();
    }

    private class DrawingPanel extends JPanel {

        private final int gridSize = 28;
        private final int scale = 20;
        private final int[][] pixels;
        private DrawingListener drawingListener;

        public DrawingPanel() {
            pixels = new int[gridSize][gridSize];
            setBackground(Color.WHITE);

            addMouseListener(new MouseAdapter() {
                public void mousePressed(MouseEvent e) {
                    handleDrawing(e);
                }
            });
            addMouseMotionListener(new MouseAdapter() {
                public void mouseDragged(MouseEvent e) {
                    handleDrawing(e);
                }
            });
        }

        public void setDrawingListener(DrawingListener listener) {
            this.drawingListener = listener;
        }

        private void handleDrawing(MouseEvent e) {
            int x = e.getX() / scale;
            int y = e.getY() / scale;
            if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
                // Se è selezionata la penna, impostiamo il pixel a 1, altrimenti (gomma) a 0.
                if (isPenSelected()) {
                    pixels[y][x] = 1;
                } else {
                    pixels[y][x] = 0;
                }
                repaint();
                if (drawingListener != null) {
                    drawingListener.onDrawingChanged();
                }
            }
        }

        public double[] getImageData() {
            double[] data = new double[gridSize * gridSize];
            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) {
                    data[y * gridSize + x] = pixels[y][x];
                }
            }
            return data;
        }

        public void clear() {
            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) {
                    pixels[y][x] = 0;
                }
            }
            repaint();
        }

        public int getScale() {
            return scale;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) {
                    g.setColor(pixels[y][x] == 1 ? Color.BLACK : Color.WHITE);
                    g.fillRect(x * scale, y * scale, scale, scale);
                    g.setColor(Color.LIGHT_GRAY);
                    g.drawRect(x * scale, y * scale, scale, scale);
                }
            }
        }
    }

    private interface DrawingListener {
        void onDrawingChanged();
    }

    private static class PredictionPanel extends JPanel {
        private final JLabel predictionLabel;
        private final JTextArea distributionArea;

        public PredictionPanel() {
            setLayout(new BorderLayout());
            predictionLabel = new JLabel("Predetto: -", SwingConstants.CENTER);
            predictionLabel.setFont(new Font("Arial", Font.BOLD, 24));
            distributionArea = new JTextArea();
            distributionArea.setEditable(false);
            distributionArea.setFont(new Font("Monospaced", Font.PLAIN, 14));
            add(predictionLabel, BorderLayout.NORTH);
            add(new JScrollPane(distributionArea), BorderLayout.CENTER);
        }

        public void updatePrediction(int predictedDigit, Vector distribution) {
            predictionLabel.setText("Predetto: " + predictedDigit);
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < distribution.size(); i++) {
                sb.append(i).append(": ")
                        .append(String.format("%.3f", distribution.get(i)))
                        .append("\n");
            }
            distributionArea.setText(sb.toString());
        }
    }

    public static void main(String[] args) {
        Model model = new Model();
        model.load("mnist-99.json");
        model.reloadMatrices();
        SwingUtilities.invokeLater(() -> new DigitRecognizerApp(model));
    }
}
