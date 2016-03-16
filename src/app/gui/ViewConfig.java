/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * 
 * This file is part of MultilayerPerceptron.
 * 
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.gui;

import java.util.Collections;
import java.util.Vector;

import javax.swing.JDialog;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;

import app.Config;
import app.Translate;

public class ViewConfig extends JDialog {

	private static final long serialVersionUID = 6795269672488995634L;

	public ViewConfig() {
		setSize(500, 300);
		setTitle(Config.get("APP_NAME"));
		JScrollPane panel = new JScrollPane();
		add(panel);
		JTable table = new JTable();
		panel.setViewportView(table);
		new ConfigModelTable(table);
		setLocationRelativeTo(this);
		setModal(true);
		setVisible(true);
	}

	public class ConfigModelTable extends AbstractTableModel {

		private static final long serialVersionUID = -8276891304790814037L;
		private Vector<String> titles;
		private Vector<String> keys;
		private JTable table;

		public ConfigModelTable(JTable table) {
			titles = new Vector<String>();
			titles.add("#");
			titles.add(Translate.get("GUI_NAME"));
			titles.add(Translate.get("GUI_VALUE"));
			keys = Config.getKeys();
			Collections.sort(keys);

			this.table = table;
			this.table.setModel(this);
			this.table.getColumnModel().getColumn(0).setMaxWidth(20);
		}

		@Override
		public int getColumnCount() {
			return titles.size();
		}

		@Override
		public int getRowCount() {
			return keys.size();
		}

		@Override
		public Object getValueAt(int rowIndex, int columnIndex) {
			switch (columnIndex) {
			case 0:
				return rowIndex + 1;
			case 1:
				return keys.get(rowIndex);
			case 2:
				return Config.get(keys.get(rowIndex));
			}
			return null;
		}

		@Override
		public String getColumnName(int column) {
			return titles.get(column);
		}

	}
}
