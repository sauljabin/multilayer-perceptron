/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * 
 * This file is part of MultilayerPerceptron.
 * 
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app;

import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

import org.jvnet.substance.SubstanceLookAndFeel;
import org.jvnet.substance.skin.BusinessBlueSteelSkin;

import app.gui.ControllerViewApp;

public class Main {
	
	private static void loadFeatures() {
		try {
			Config.load();
			Config.save();
			Translate.load();
		} catch (Exception e) {
			Log.error(Main.class, "loadFeatures()", e);
			System.exit(0);
		}
	}

	public static void main(String[] args) {
		loadFeatures();
		JFrame.setDefaultLookAndFeelDecorated(true);
		JDialog.setDefaultLookAndFeelDecorated(true);
		SubstanceLookAndFeel.setSkin(new BusinessBlueSteelSkin());
		UIManager.put(SubstanceLookAndFeel.BUTTON_NO_MIN_SIZE_PROPERTY, Boolean.TRUE);
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				new ControllerViewApp();
			}
		});
	}

}
