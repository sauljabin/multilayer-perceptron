/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.gui;

import app.Log;
import app.Translate;
import app.perceptron.MultilayerPerceptron;
import app.util.UtilFileText;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class ControllerViewApp extends WindowAdapter implements ActionListener, ChangeListener, Runnable {

    private ViewApp viewApp;
    private Thread threadTraining;
    private MultilayerPerceptron perceptron;

    public ControllerViewApp() {
        viewApp = new ViewApp();
        viewApp.setController(this);
        viewApp.getPathResults().setText("documents/AND_Results.txt");
        viewApp.getPathFileTraining().setText("documents/AND_TrainingValues.txt");
        viewApp.getPathTestValues().setText("documents/AND_TestValues.txt");
        viewApp.getTxtName().setText("AND");
        Log.setLogTextArea(viewApp.getTarConsole());
    }

    @Override
    public void windowClosing(WindowEvent e) {
        close();
    }

    public void close() {
        viewApp.dispose();
        System.exit(0);
    }

    public void about() {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ViewAbout();
            }
        });
    }

    public void showConfig() {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ViewConfig();
            }
        });
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Object source = e.getSource();
        if (source.equals(viewApp.getMenuItemClose()))
            close();
        else if (source.equals(viewApp.getMenuItemAbout()))
            about();
        else if (source.equals(viewApp.getMenuItemShowConfig()))
            showConfig();
        else if (source.equals(viewApp.getBtnClose()))
            close();
        else if (source.equals(viewApp.getBtnTraining()))
            training();
        else if (source.equals(viewApp.getBtnPathFileTraining()))
            getPathFileTraining();
        else if (source.equals(viewApp.getBtnPathResults()))
            getPathResults();
        else if (source.equals(viewApp.getBtnPathTestValues()))
            getPathTestValues();
    }

    public File fileChooser(String path) {
        JFileChooser file = new JFileChooser(new File(path));
        file.showDialog(viewApp, Translate.get("GUI_OPEN"));
        return file.getSelectedFile();
    }

    public void getPathResults() {
        String path = viewApp.getPathResults().getText();
        if (path.trim().isEmpty())
            path = ".";
        File file = fileChooser(path);
        if (file != null) {
            viewApp.getPathResults().setText(file.getAbsolutePath());
        }
    }

    public void getPathFileTraining() {
        String path = viewApp.getPathFileTraining().getText();
        if (path.trim().isEmpty())
            path = ".";
        File file = fileChooser(path);
        if (file != null) {
            viewApp.getPathFileTraining().setText(file.getAbsolutePath());
        }
    }

    private void getPathTestValues() {
        String path = viewApp.getPathTestValues().getText();
        if (path.trim().isEmpty())
            path = ".";
        File file = fileChooser(path);
        if (file != null) {
            viewApp.getPathTestValues().setText(file.getAbsolutePath());
        }

    }

    public void training() {

        if (viewApp.getTxtName().getText().trim().isEmpty()) {
            JOptionPane.showMessageDialog(viewApp, Translate.get("ERROR_NAMEEMPTY"), Translate.get("GUI_ERROR"), JOptionPane.ERROR_MESSAGE);
            return;
        }

        if (viewApp.getPathTestValues().getText().trim().isEmpty() || viewApp.getPathFileTraining().getText().trim().isEmpty() || viewApp.getPathResults().getText().trim().isEmpty()) {
            JOptionPane.showMessageDialog(viewApp, Translate.get("ERROR_PATHEMPTY"), Translate.get("GUI_ERROR"), JOptionPane.ERROR_MESSAGE);
            return;
        }

        viewApp.getTxtName().setEditable(false);
        viewApp.getPathFileTraining().setEnabled(false);
        viewApp.getPathResults().setEnabled(false);
        viewApp.getPathTestValues().setEnabled(false);
        viewApp.getBtnPathFileTraining().setEnabled(false);
        viewApp.getBtnPathResults().setEnabled(false);
        viewApp.getBtnTraining().setEnabled(false);
        viewApp.getBtnPathTestValues().setEnabled(false);
        viewApp.getSpiHiddenLayerSize().setEnabled(false);
        viewApp.getSpiLearningRate().setEnabled(false);
        viewApp.getSpiMaxError().setEnabled(false);
        viewApp.getSpiMaxPeriods().setEnabled(false);
        viewApp.getSpiVariationRate().setEnabled(false);

        threadTraining = new Thread(this);
        threadTraining.start();
    }

    public void enable() {
        viewApp.getTxtName().setEditable(true);
        viewApp.getPathFileTraining().setEnabled(true);
        viewApp.getPathResults().setEnabled(true);
        viewApp.getPathTestValues().setEnabled(true);
        viewApp.getBtnPathFileTraining().setEnabled(true);
        viewApp.getBtnPathResults().setEnabled(true);
        viewApp.getBtnTraining().setEnabled(true);
        viewApp.getBtnPathTestValues().setEnabled(true);
        viewApp.getSpiHiddenLayerSize().setEnabled(true);
        viewApp.getSpiLearningRate().setEnabled(true);
        viewApp.getSpiMaxError().setEnabled(true);
        viewApp.getSpiMaxPeriods().setEnabled(true);
        viewApp.getSpiVariationRate().setEnabled(true);
    }

    @Override
    public void stateChanged(ChangeEvent e) {

    }

    @Override
    public void run() {
        UtilFileText uft = new UtilFileText();
        String pathFileTraining = viewApp.getPathFileTraining().getText();
        String pathTestValues = viewApp.getPathTestValues().getText();
        String pathResults = viewApp.getPathResults().getText();
        String name = viewApp.getTxtName().getText();
        double learningRate = (double) viewApp.getSpiLearningRate().getValue();
        double variationRate = (double) viewApp.getSpiVariationRate().getValue();
        double maxError = (double) viewApp.getSpiMaxError().getValue();
        int maxPeriods = (int) viewApp.getSpiMaxPeriods().getValue();
        int hiddenLayerSize = (int) viewApp.getSpiHiddenLayerSize().getValue();
        List<String> trainingValuesList;
        List<String> testValuesList;
        double[][] trainingValues;
        double[][] desiredOutput;
        double[][] testValues;

        try {
            trainingValuesList = uft.readFileToList(pathFileTraining);
            trainingValues = new double[trainingValuesList.size()][];
            desiredOutput = new double[trainingValuesList.size()][];
        } catch (IOException e) {
            enable();
            Log.error(ControllerViewApp.class, Translate.get("ERROR_LOADTRAININGVALUES"), e);
            return;
        }

        for (int i = 0; i < trainingValuesList.size(); i++) {
            String s = trainingValuesList.get(i);

            String[] string = s.split(";");

            String[] inputString = string[0].split(",");
            double[] input = new double[inputString.length];

            String[] outputString = string[1].split(",");
            double[] output = new double[outputString.length];

            for (int j = 0; j < input.length; j++) {
                input[j] = Double.parseDouble(inputString[j]);
            }

            for (int j = 0; j < output.length; j++) {
                output[j] = Double.parseDouble(outputString[j]);
            }

            trainingValues[i] = input;
            desiredOutput[i] = output;
        }

        try {
            testValuesList = uft.readFileToList(pathTestValues);
            testValues = new double[testValuesList.size()][];
        } catch (IOException e) {
            enable();
            Log.error(ControllerViewApp.class, Translate.get("ERROR_LOADTESTVALUES"), e);
            return;
        }

        for (int i = 0; i < testValuesList.size(); i++) {
            String s = testValuesList.get(i);
            String[] inputString = s.split(",");
            double[] input = new double[inputString.length];
            for (int j = 0; j < input.length; j++) {
                input[j] = Double.parseDouble(inputString[j]);
            }
            testValues[i] = input;
        }

        Log.info(ControllerViewApp.class, Translate.get("INFO_INITTRAINING"));

        perceptron = new MultilayerPerceptron(trainingValues[0].length, hiddenLayerSize, desiredOutput[0].length, learningRate, variationRate, maxPeriods, maxError);
        perceptron.training(trainingValues, desiredOutput);
        Log.info(ControllerViewApp.class, Translate.get("INFO_FINISHTRAINING"));

        String print = Translate.get("GUI_NAME") + ": " + name;
        print += "\n" + Translate.get("GUI_LEARNINGRATE") + ": " + perceptron.getLearningRate();
        print += "\n" + Translate.get("GUI_VARIATIONRATE") + ": " + perceptron.getVariationRate();
        print += "\n" + Translate.get("GUI_PERIODS") + ": " + perceptron.getPeriods();
        print += "\n" + Translate.get("GUI_HIDDENLAYERSIZE") + ": " + perceptron.getHiddenLayerSize();
        print += "\n\n" + Translate.get("GUI_PERIODS") + "\t" + Translate.get("GUI_ERROR");

        for (int i = 0; i < perceptron.getErrors().size(); i++) {
            print += "\n" + i;
            print += "\t" + perceptron.getErrors().get(i);
        }

        print += "\n\n" + Translate.get("GUI_TESTVALUES") + "\n";
        print += Translate.get("GUI_INPUT") + "\t" + Translate.get("GUI_OUTPUT");

        for (int i = 0; i < testValues.length; i++) {
            print += "\n";
            for (int j = 0; j < testValues[i].length; j++) {
                print += testValues[i][j] + " ";
            }
            print += "\t";

            double[] output = perceptron.output(testValues[i]);

            for (int j = 0; j < output.length; j++) {
                print += output[j] + " ";
            }
        }

        Log.info(ControllerViewApp.class, Translate.get("INFO_SAVERESULTS"));
        try {
            uft.writeFile(pathResults, print);
        } catch (Exception e) {
            Log.error(ControllerViewApp.class, Translate.get("ERROR_SAVERESULTS"), e);
        }

        ViewGraphic viewG = new ViewGraphic(name);

        for (int i = 0; i < perceptron.getErrors().size(); i++) {
            viewG.addPoint(i, perceptron.getErrors().get(i));
        }
        try {
            viewG.exportImage(viewApp.getPathResults().getText() + ".png");
        } catch (IOException e) {
            Log.error(ControllerViewApp.class, Translate.get("ERROR_SAVERESULTSIMAGE"), e);
        }

        enable();
    }

}
