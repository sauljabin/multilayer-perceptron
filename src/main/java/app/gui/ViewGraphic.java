/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.gui;

import app.Translate;
import app.util.UtilImage;
import net.miginfocom.swing.MigLayout;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;

public class ViewGraphic extends JDialog {

    private XYSeriesCollection dataset;
    private JFreeChart chart;
    private XYSeries serieError;

    private JLabel lblError;

    public ViewGraphic(String title) {
        setSize(600, 400);
        setLocationRelativeTo(this);
        setTitle(title);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent arg0) {
                dispose();
            }
        });
        setLayout(new BorderLayout());

        serieError = new XYSeries(title);

        dataset = new XYSeriesCollection();
        dataset.addSeries(serieError);

        chart = ChartFactory.createXYLineChart(title, Translate.get("GUI_PERIODS"), Translate.get("GUI_ERROR"), dataset, PlotOrientation.VERTICAL, true, true, false);
        add(new ChartPanel(chart), BorderLayout.CENTER);

        JPanel south = new JPanel(new MigLayout());

        lblError = new JLabel("");
        lblError.setHorizontalTextPosition(SwingConstants.LEFT);
        lblError.setFont(new Font("ARIAL", Font.BOLD, 26));
        JLabel lblErrorTitle = new JLabel(Translate.get("GUI_ERROR") + ": ");
        lblErrorTitle.setFont(new Font("ARIAL", Font.BOLD, 26));
        south.add(lblErrorTitle);
        south.add(lblError, "wrap");

        add(south, BorderLayout.SOUTH);
        setVisible(true);
    }

    public void addPoint(int period, double error) {
        serieError.add(period, error);
        lblError.setText(String.format("%f", error));
    }

    public void exportImage(String path) throws IOException {
        UtilImage ui = new UtilImage();
        ui.writeImage(chart.createBufferedImage(600, 400), path);
    }

}
