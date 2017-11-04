/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.gui;

import app.Config;
import app.Translate;
import net.miginfocom.swing.MigLayout;

import javax.swing.*;
import javax.swing.text.DefaultCaret;
import java.awt.*;
import java.util.Vector;

public class ViewApp extends JFrame {

    private ControllerViewApp controller;
    private Vector<JMenuItem> menuItems;
    private Vector<JButton> buttons;
    private Vector<JSpinner> sppiners;
    private JMenuBar menuBar;
    private JMenu menuOptions;
    private JMenuItem menuItemShowConfig;
    private JMenuItem menuItemClose;
    private JMenu menuHelp;
    private JMenuItem menuItemAbout;
    private JPanel pnlSouth;
    private JTextArea tarConsole;
    private JPanel pnlCentral;
    private JTextField pathFileTraining;
    private JButton btnPathFileTraining;
    private JTextField pathResults;
    private JButton btnPathResults;
    private JSpinner spiLearningRate;
    private JSpinner spiMaxPeriods;
    private JPanel pnlLateral;
    private JButton btnTraining;
    private JButton btnClose;
    private JTextField txtName;
    private JTextField pathTestValues;
    private JButton btnPathTestValues;

    private JSpinner spiVariationRate;

    private JSpinner spiMaxError;

    private JSpinner spiHiddenLayerSize;

    public ViewApp() {
        menuItems = new Vector<JMenuItem>();
        buttons = new Vector<JButton>();
        sppiners = new Vector<JSpinner>();
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        init();
        setLocationRelativeTo(this);
        setVisible(true);
    }

    private void init() {
        setLayout(new BorderLayout());
        setSize(600, 450);
        setTitle(Config.get("APP_NAME"));

        menuBar = new JMenuBar();
        setJMenuBar(menuBar);

        menuOptions = new JMenu(Translate.get("GUI_OPTIONS"));
        menuBar.add(menuOptions);

        menuItemShowConfig = new JMenuItem(Translate.get("GUI_SHOWCONFIG"));
        menuOptions.add(menuItemShowConfig);

        menuItemClose = new JMenuItem(Translate.get("GUI_CLOSE"));
        menuOptions.add(menuItemClose);

        menuHelp = new JMenu(Translate.get("GUI_HELP"));
        menuBar.add(menuHelp);

        menuItemAbout = new JMenuItem(Translate.get("GUI_ABOUT"));
        menuHelp.add(menuItemAbout);

        pnlCentral = new JPanel();
        pnlCentral.setLayout(new MigLayout());
        add(pnlCentral, BorderLayout.CENTER);

        pnlCentral.add(new JLabel(Translate.get("GUI_NAME")));
        txtName = new JTextField();
        pnlCentral.add(txtName, "width 300, height 25, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_TRAININGSVALUE")));
        pathFileTraining = new JTextField();
        pnlCentral.add(pathFileTraining, "width 300, height 25");
        btnPathFileTraining = new JButton("...");
        pnlCentral.add(btnPathFileTraining, "width 20, height 25, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_TESTVALUES")));
        pathTestValues = new JTextField();
        pnlCentral.add(pathTestValues, "width 300, height 25");
        btnPathTestValues = new JButton("...");
        pnlCentral.add(btnPathTestValues, "width 20, height 25, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_RESULTS")));
        pathResults = new JTextField();
        pnlCentral.add(pathResults, "width 300, height 25");
        btnPathResults = new JButton("...");
        pnlCentral.add(btnPathResults, "width 20, height 25, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_HIDDENLAYERSIZE")));
        spiHiddenLayerSize = new JSpinner();
        spiHiddenLayerSize.setModel(new SpinnerNumberModel(5, 1, 1000, 1));
        pnlCentral.add(spiHiddenLayerSize, "width 100, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_LEARNINGRATE")));
        spiLearningRate = new JSpinner();
        spiLearningRate.setModel(new SpinnerNumberModel(0.01, 0.0, 1.0, 0.001));
        pnlCentral.add(spiLearningRate, "width 100, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_VARIATIONRATE")));
        spiVariationRate = new JSpinner();
        spiVariationRate.setModel(new SpinnerNumberModel(0.01, 0.0, 1.0, 0.001));
        pnlCentral.add(spiVariationRate, "width 100, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_MAXPERIODS")));
        spiMaxPeriods = new JSpinner();
        spiMaxPeriods.setModel(new SpinnerNumberModel(1000, 1, 100000, 1));
        pnlCentral.add(spiMaxPeriods, "width 100, wrap");

        pnlCentral.add(new JLabel(Translate.get("GUI_MAXERROR")));
        spiMaxError = new JSpinner();
        spiMaxError.setModel(new SpinnerNumberModel(0.01, 0.0, 1.0, 0.001));
        pnlCentral.add(spiMaxError, "width 100, wrap");

        pnlLateral = new JPanel();
        pnlLateral.setLayout(new MigLayout());
        add(pnlLateral, BorderLayout.EAST);

        btnTraining = new JButton(Translate.get("GUI_TRAINING"));
        pnlLateral.add(btnTraining, "width 100, height 50, wrap");
        btnClose = new JButton(Translate.get("GUI_CLOSE"));
        pnlLateral.add(btnClose, "width 100, height 50, wrap");

        pnlSouth = new JPanel();
        pnlSouth.setLayout(new MigLayout());
        add(pnlSouth, BorderLayout.SOUTH);

        JScrollPane scrollPanelConsole = new JScrollPane();
        pnlSouth.add(scrollPanelConsole, "width 100%, height 100");
        tarConsole = new JTextArea();
        DefaultCaret caret = (DefaultCaret) tarConsole.getCaret();
        caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
        scrollPanelConsole.setViewportView(tarConsole);

        menuItems.add(menuItemShowConfig);
        menuItems.add(menuItemClose);
        menuItems.add(menuItemAbout);
        buttons.add(btnClose);
        buttons.add(btnPathFileTraining);
        buttons.add(btnPathResults);
        buttons.add(btnTraining);
        buttons.add(btnPathTestValues);

    }

    public JSpinner getSpiHiddenLayerSize() {
        return spiHiddenLayerSize;
    }

    public JMenuItem getMenuItemShowConfig() {
        return menuItemShowConfig;
    }

    public JMenuItem getMenuItemClose() {
        return menuItemClose;
    }

    public JMenuItem getMenuItemAbout() {
        return menuItemAbout;
    }

    public JTextArea getTarConsole() {
        return tarConsole;
    }

    public ControllerViewApp getController() {
        return controller;
    }

    public void setController(ControllerViewApp controller) {
        this.controller = controller;
        for (int i = 0; i < menuItems.size(); i++) {
            menuItems.get(i).addActionListener(controller);
        }
        for (int i = 0; i < buttons.size(); i++) {
            buttons.get(i).addActionListener(controller);
        }
        for (int i = 0; i < sppiners.size(); i++) {
            sppiners.get(i).addChangeListener(controller);
        }
        this.addWindowListener(controller);
    }

    public JTextField getPathFileTraining() {
        return pathFileTraining;
    }

    public JButton getBtnPathFileTraining() {
        return btnPathFileTraining;
    }

    public JTextField getPathResults() {
        return pathResults;
    }

    public JButton getBtnPathResults() {
        return btnPathResults;
    }

    public JSpinner getSpiVariationRate() {
        return spiVariationRate;
    }

    public JSpinner getSpiMaxError() {
        return spiMaxError;
    }

    public JSpinner getSpiLearningRate() {
        return spiLearningRate;
    }

    public JSpinner getSpiMaxPeriods() {
        return spiMaxPeriods;
    }

    public JButton getBtnTraining() {
        return btnTraining;
    }

    public JButton getBtnClose() {
        return btnClose;
    }

    public JTextField getTxtName() {
        return txtName;
    }

    public void setTxtName(JTextField txtName) {
        this.txtName = txtName;
    }

    public JTextField getPathTestValues() {
        return pathTestValues;
    }

    public JButton getBtnPathTestValues() {
        return btnPathTestValues;
    }

}
