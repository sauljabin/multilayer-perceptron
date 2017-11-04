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

public class ViewAbout extends JDialog {

    public ViewAbout() {
        setSize(300, 170);
        setTitle(Config.get("APP_NAME"));

        JPanel panel = new JPanel(new MigLayout());
        add(panel);

        panel.add(new JLabel(Config.get("APP_NAME")), "width 300, wrap");
        panel.add(new JLabel(Translate.get("GUI_APPDESCRIP")), "grow, wrap 20");
        panel.add(new JLabel(Config.get("APP_LICENSE")), "grow, wrap");
        panel.add(new JLabel(Config.get("APP_AUTHOR")), "grow, wrap");

        setLocationRelativeTo(this);
        setModal(true);
        setVisible(true);
    }

}
