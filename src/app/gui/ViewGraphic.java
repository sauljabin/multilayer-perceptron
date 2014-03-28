/**
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 * 
 *		SAUL PIÑA - SAULJP07@GMAIL.COM
 *		2014
 */

package app.gui;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;

import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import net.miginfocom.swing.MigLayout;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import app.Translate;
import app.util.UtilImage;

public class ViewGraphic extends JDialog {

	private static final long serialVersionUID = -7690162952744313595L;

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