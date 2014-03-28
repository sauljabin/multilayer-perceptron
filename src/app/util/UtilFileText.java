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
 */

package app.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for reading and writing text files
 * 
 * @author Saul Pina - sauljp07@gmail.com
 * @version 1.0.0
 */
public class UtilFileText {

	public static final String VERSION = "1.0.0";

	public static final String CHARSET_UTF_8 = "UTF-8";
	public static final String CHARSET_ISO_8859_1 = "ISO-8859-1";

	private File file;
	private String charset;

	/**
	 * Get path
	 * 
	 * @return Path
	 */
	public String getPath() {
		return file.getPath();
	}

	/**
	 * Set file path
	 * 
	 * @param path
	 *            File path
	 */
	public void setPath(String path) {
		this.file = new File(path);
	}

	/**
	 * Get file
	 * 
	 * @return File
	 */
	public File getFile() {
		return file;
	}

	/**
	 * Set file
	 * 
	 * @param file
	 *            File
	 */
	public void setFile(File file) {
		this.file = file;
	}

	/**
	 * Get charset
	 * 
	 * @return Charset
	 */
	public String getCharset() {
		return charset;
	}

	/**
	 * Set the charset
	 * 
	 * @param charset
	 *            Charset, UtilFileText.CHARSET_UTF_8
	 */
	public void setCharset(String charset) {
		this.charset = charset;
	}

	/**
	 * @param file
	 *            File
	 * @param charset
	 *            Charset
	 */
	public UtilFileText(File file, String charset) {
		this.file = file;
		this.charset = charset;
	}

	/**
	 * @param path
	 *            File path
	 * @param charset
	 *            Charset
	 */
	public UtilFileText(String path, String charset) {
		this(new File(path), charset);
	}

	/**
	 * @param path
	 *            File path
	 */
	public UtilFileText(String path) {
		this(new File(path), CHARSET_UTF_8);
	}

	/**
	 * @param file
	 *            File
	 */
	public UtilFileText(File file) {
		this(file, CHARSET_UTF_8);
	}

	public UtilFileText() {
		this.file = null;
		this.charset = CHARSET_UTF_8;
	}

	/**
	 * Read a text file
	 * 
	 * @return Text
	 * @throws IOException
	 */
	public String readFile() throws IOException {
		return readFile(file, charset);
	}

	/**
	 * Read a text file
	 * 
	 * @return List of lines
	 * @throws IOException
	 */
	public List<String> readFileToList() throws IOException {
		return readFileToList(file, charset);
	}

	/**
	 * Read a text file
	 * 
	 * @param path
	 *            File path
	 * @return Text
	 * @throws IOException
	 */
	public String readFile(String path) throws IOException {
		return readFile(new File(path), CHARSET_UTF_8);
	}

	/**
	 * Read a text file
	 * 
	 * @param path
	 *            File path
	 * @return List of lines
	 * @throws IOException
	 */
	public List<String> readFileToList(String path) throws IOException {
		return readFileToList(new File(path), CHARSET_UTF_8);
	}

	/**
	 * Read a text file
	 * 
	 * @param file
	 *            File
	 * @return Text
	 * @throws IOException
	 */
	public String readFile(File file) throws IOException {
		return readFile(file, CHARSET_UTF_8);
	}

	/**
	 * Read a text file
	 * 
	 * @param file
	 *            File
	 * @return List of lines
	 * @throws IOException
	 */
	public List<String> readFileToList(File file) throws IOException {
		return readFileToList(file, CHARSET_UTF_8);
	}

	/**
	 * Read a text file
	 * 
	 * @param path
	 *            File path
	 * @param charset
	 *            Charset
	 * @return Text
	 * @throws IOException
	 */
	public String readFile(String path, String charset) throws IOException {
		return readFile(new File(path), charset);
	}

	/**
	 * Read a text file
	 * 
	 * @param path
	 *            File path
	 * @param charset
	 *            Charset
	 * @return List of lines
	 * @throws IOException
	 */
	public List<String> readFileToList(String path, String charset) throws IOException {
		return readFileToList(new File(path), charset);
	}

	/**
	 * Read a text file
	 * 
	 * @param file
	 *            File
	 * @param charset
	 *            Charset
	 * @return Text
	 * @throws IOException
	 */
	public String readFile(File file, String charset) throws IOException {
		String line;
		StringBuilder buffer = new StringBuilder();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), charset));
		while ((line = br.readLine()) != null) {
			buffer.append(line + "\n");
		}
		br.close();
		return buffer.toString();
	}

	/**
	 * Read a text file
	 * 
	 * @param file
	 *            File
	 * @param charset
	 *            Charset
	 * @return List of lines
	 * @throws IOException
	 */
	public List<String> readFileToList(File file, String charset) throws IOException {
		List<String> list = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), charset));
		String line;
		while ((line = br.readLine()) != null) {
			list.add(line);
		}
		br.close();
		return list;
	}

	/**
	 * Write a text file
	 * 
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(String text) throws IOException {
		writeFile(file, text, charset);
	}

	/**
	 * Write a text file
	 * 
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(List<String> text) throws IOException {
		writeFile(file, text, charset);
	}

	/**
	 * Write a text file
	 * 
	 * @param file
	 *            File
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(File file, String text) throws IOException {
		writeFile(file, text, CHARSET_UTF_8);
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(String path, String text) throws IOException {
		writeFile(new File(path), text, CHARSET_UTF_8);
	}

	/**
	 * Write a text file
	 * 
	 * @param file
	 *            File
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(File file, List<String> text) throws IOException {
		writeFile(file, text, CHARSET_UTF_8);
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void writeFile(String path, List<String> text) throws IOException {
		writeFile(new File(path), text, CHARSET_UTF_8);
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void writeFile(String path, String text, String charset) throws IOException {
		writeFile(new File(path), text, charset);
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void writeFile(File file, String text, String charset) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), charset));
		out.write(text);
		out.flush();
		out.close();
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void writeFile(String path, List<String> text, String charset) throws IOException {
		writeFile(new File(path), text, charset);
	}

	/**
	 * Write a text file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void writeFile(File file, List<String> text, String charset) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), charset));
		for (String line : text) {
			out.write(line + "\n");
		}
		out.flush();
		out.close();
	}

	/**
	 * Append a text to file
	 * 
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(String text) throws IOException {
		appendFile(file, text, charset);
	}

	/**
	 * Append a text to file
	 * 
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(List<String> text) throws IOException {
		appendFile(file, text, charset);
	}

	/**
	 * Append a text to file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void appendFile(String path, String text, String charset) throws IOException {
		appendFile(new File(path), text, charset);
	}

	/**
	 * Append a text to file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void appendFile(String path, List<String> text, String charset) throws IOException {
		appendFile(new File(path), text, charset);
	}

	/**
	 * Append a text to file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(String path, String text) throws IOException {
		appendFile(new File(path), text, CHARSET_UTF_8);
	}

	/**
	 * Append a text to file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(String path, List<String> text) throws IOException {
		appendFile(new File(path), text, CHARSET_UTF_8);
	}

	/**
	 * Append a text to file
	 * 
	 * @param file
	 *            File
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(File file, String text) throws IOException {
		appendFile(file, text, CHARSET_UTF_8);
	}

	/**
	 * Append a text to file
	 * 
	 * @param path
	 *            File path
	 * @param text
	 *            Text to write
	 * @throws IOException
	 */
	public void appendFile(File file, List<String> text) throws IOException {
		appendFile(file, text, CHARSET_UTF_8);
	}

	/**
	 * Append a text to file
	 * 
	 * @param file
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void appendFile(File file, String text, String charset) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), charset));
		out.write(text);
		out.flush();
		out.close();
	}

	/**
	 * Append a text to file
	 * 
	 * @param file
	 *            File path
	 * @param text
	 *            Text to write
	 * @param charset
	 *            Charset
	 * @throws IOException
	 */
	public void appendFile(File file, List<String> text, String charset) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), charset));
		for (String line : text) {
			out.write(line + "\n");
		}
		out.flush();
		out.close();
	}
}
