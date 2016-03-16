/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * 
 * This file is part of MultilayerPerceptron.
 * 
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.PrintWriter;

import javax.swing.JTextArea;

import app.util.UtilDate;

public class Log {

	public static final int NONE = 2;
	public static final int INFO = 4;
	public static final int WARN = 8;
	public static final int ERROR = 16;
	public static final int ALL = 32;

	private static int nivel = ALL;
	private static JTextArea textArea;

	public static int getNivel() {
		return nivel;
	}

	/**
	 * Set nivel for print: UtilLog.NONE, UtilLog.INFO, UtilLog.WARN,
	 * UtilLog.ERROR, UtilLog.ALL
	 * 
	 * @param nivel
	 */
	public static void setNivel(int nivel) {
		Log.nivel = nivel;
	}

	private static void print(int printNivel, Class<?> clazz, String msg, Exception e) {
		if (nivel == NONE)
			return;

		if (printNivel > nivel)
			return;

		String type = "";

		PrintStream ps = System.out;

		switch (printNivel) {
		case INFO:
			type = "INFO";
			break;
		case WARN:
			type = "WARN";
			break;
		case ERROR:
			type = "ERROR";
			break;
		default:
			break;
		}

		String string = String.format("TYPE: %s\nNAME: %s\nTIME: %s\n----> %s\n", type, clazz.getName(), UtilDate.nowFormat("yyyy-MM-dd HH:mm"), msg);
		ps.print(string);
		if (e != null) {
			e.printStackTrace(ps);
		}
		ps.println();

		File folder = new File("log");
		folder.mkdir();
		FileOutputStream fs;
		OutputStreamWriter os;
		BufferedWriter bw;
		try {
			fs = new FileOutputStream(folder.getPath() + "/" + UtilDate.nowFormat("yyyy-MM-dd") + ".log", true);
			os = new OutputStreamWriter(fs, "UTF-8");
			bw = new BufferedWriter(os);
			bw.write(string);
			bw.flush();
			if (e != null)
				e.printStackTrace(new PrintWriter(os));
			bw.write("\n");
			bw.close();
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		if (textArea != null)
			textArea.append(string + "\n");

	}

	public static void info(Class<?> clazz, String msg) {
		print(INFO, clazz, msg, null);
	}

	public static void info(Class<?> clazz, String msg, Exception e) {
		print(INFO, clazz, msg, e);
	}

	public static void error(Class<?> clazz, String msg) {
		print(ERROR, clazz, msg, null);
	}

	public static void error(Class<?> clazz, String msg, Exception e) {
		print(ERROR, clazz, msg, e);
	}

	public static void warning(Class<?> clazz, String msg) {
		print(WARN, clazz, msg, null);
	}

	public static void warning(Class<?> clazz, String msg, Exception e) {
		print(WARN, clazz, msg, e);
	}

	public static void setLogTextArea(JTextArea textArea) {
		Log.textArea = textArea;
		Log.textArea.setEditable(false);
	}

}
