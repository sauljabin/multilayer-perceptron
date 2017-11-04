/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app;

import app.gui.ControllerViewApp;

import javax.swing.*;

public class Main {

    private static void loadFeatures() {
        try {
            Config.load();
            Translate.load();
        } catch (Exception e) {
            Log.error(Main.class, "loadFeatures()", e);
            System.exit(0);
        }
    }

    public static void main(String[] args) {
        loadFeatures();
        SwingUtilities.invokeLater(() -> new ControllerViewApp());
    }

}
