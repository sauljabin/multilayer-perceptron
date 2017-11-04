/**
 * Copyright (c) 2014 Saúl Piña <sauljabin@gmail.com>.
 * <p>
 * This file is part of MultilayerPerceptron.
 * <p>
 * MultilayerPerceptron is licensed under The MIT License.
 * For full copyright and license information please see the LICENSE file.
 */

package app.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

/**
 * Image Handling
 *
 * @author Saul Pina - sauljp07@gmail.com
 * @version 1.0.0
 */
public class UtilImage {

    public static final String VERSION = "1.0.0";

    /**
     * Read an image from disk
     *
     * @param path File path
     * @return BufferedImage image format
     * @throws IOException
     */
    public BufferedImage readImage(String path) throws IOException {
        return readImage(new File(path));
    }

    /**
     * Read an image from disk
     *
     * @param file File
     * @return BufferedImage image format
     * @throws IOException
     */
    public BufferedImage readImage(File file) throws IOException {
        return ImageIO.read(file);
    }

    /**
     * Write image to disk
     *
     * @param image Image writing
     * @param path  File path
     * @return True if was written successfully
     * @throws IOException
     */
    public boolean writeImage(BufferedImage image, String path) throws IOException {
        return writeImage(image, new File(path));
    }

    /**
     * Write image to disk
     *
     * @param image Image writing
     * @param file  File
     * @return True if was written successfully
     * @throws IOException
     */
    public boolean writeImage(BufferedImage image, File file) throws IOException {
        String fileName = file.getName();
        String extension = fileName.substring(fileName.lastIndexOf('.') + 1, fileName.length());
        return ImageIO.write(image, extension, file);
    }

    /**
     * Take a BufferedImage image format and transforms it into binary
     *
     * @param image     Image transform
     * @param extension Extension of the image
     * @return Image in binary format
     * @throws IOException
     */
    public byte[] bufferedImageToByteArray(BufferedImage image, String extension) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, extension, baos);
        baos.flush();
        byte[] imageInByte = baos.toByteArray();
        baos.close();
        return imageInByte;
    }

    /**
     * Take a picture as binary array and transforms it into BufferedImage
     *
     * @param image Image in bytes
     * @return An image type BufferedImage
     * @throws IOException
     */
    public BufferedImage byteArrayToBufferedImage(byte[] image) throws IOException {
        InputStream in = new ByteArrayInputStream(image);
        return ImageIO.read(in);
    }

    /**
     * This function resizes an image without loss
     * <p>
     * Rescale the picture as a percentage
     *
     * @param image   Image
     * @param percent Percentage
     * @return Image rescaled
     */
    public BufferedImage resizePercent(BufferedImage image, float percent) {
        if (percent > 1) {
            percent = percent / 100;
        }
        int width = (int) (image.getWidth() * percent);
        int height = (int) (image.getHeight() * percent);
        return resize(image, width, height);
    }

    /**
     * This function resizes an image without loss
     * <p>
     * Take a high and provides maximum image width
     *
     * @param image  Image
     * @param height Image height
     * @return Image rescaled
     */
    public BufferedImage resizeMaxHeight(BufferedImage image, int height) {
        int width = (int) (image.getWidth() * (height * 100 / image.getHeight()) / 100);
        return resize(image, width, height);
    }

    /**
     * This function resizes an image without loss
     * <p>
     * Toma un ancho maximo y proporciona el alto de la imagen
     *
     * @param image Image
     * @param width Image width
     * @return Image rescaled
     */
    public BufferedImage resizeMaxWidth(BufferedImage image, int width) {
        int height = (int) (image.getHeight() * (width * 100 / image.getWidth()) / 100);
        return resize(image, width, height);
    }

    /**
     * This function resizes an image without loss
     *
     * @param image  Imagen
     * @param width  Image width
     * @param height Image height
     * @return Imagen rescaled
     */
    public BufferedImage resize(BufferedImage image, int width, int height) {
        int type = image.getType() == 0 ? BufferedImage.TYPE_INT_ARGB : image.getType();
        BufferedImage resizedImage = new BufferedImage(width, height, type);
        Graphics2D g = resizedImage.createGraphics();
        g.setComposite(AlphaComposite.Src);

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resizedImage;
    }
}
