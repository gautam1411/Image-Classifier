/**
 * @ Author: Yuting Liu
 *
 * This is the class for each image instance
 */

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;


public class Instance {
    // store the bufferedImage
    private BufferedImage image;
    private String label;
    private int width, height;
    // separate rgb channels
    private int[][] red_channel, green_channel, blue_channel, gray_image;

    // Constructor
    // given the bufferedimage and its class label
    // get the
    public Instance(BufferedImage image, String label) {
        this.image = image;
        this.label = label;
        width  = image.getWidth();
        height = image.getHeight();

        gray_image = null;
        // get separate rgb channels
        red_channel   = new int[height][width];
        green_channel = new int[height][width];
        blue_channel  = new int[height][width];

        for(int row = 0; row < height; ++row) {
            for(int col = 0; col < width; ++col) {
                Color c = new Color(image.getRGB(col, row));
                red_channel[  row][col] = c.getRed();
                green_channel[row][col] = c.getGreen();
                blue_channel[ row][col] = c.getBlue();
            }
        }
    }

    // get separate red channel image
    public int[][] getRedChannel() {
        return red_channel;
    }

    // get separate green channel image
    public int[][] getGreenChannel() {
        return green_channel;
    }

    // get separate blue channel image
    public int[][] getBlueChannel() {
        return blue_channel;
    }

    // get the gray scale image
    public int[][] getGrayImage() {
        // avoid repeated conversion if get the gray image before
        if(gray_image != null) {
            return gray_image;
        }

        gray_image = new int[height][width];

        // Gray filter
        for(int row = 0; row < height; ++row) {
            for(int col = 0; col < width; ++col) {
                int rgb = image.getRGB(col, row) & 0xFF;
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >>  8) & 0xFF;
                int b = (rgb        & 0xFF);
                gray_image[row][col] = (r + g + b) / 3;
            }
        }
        return gray_image;
    }



    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getLabel() {
        return label;
    }

    // display the gray-image bitmap
    public void display2D(int[][] img) {
        BufferedImage bufferedImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for(int row = 0; row < height; ++row) {
            for(int col = 0; col < width; ++col) {
                int c = img[row][col] << 16 | img[row][col] << 8 | img[row][col];
                bufferedImg.setRGB(col, row, c);
            }
        }
        // displayImage(bufferedImg);
    }

    // display the buffered image in the panel
    public void displayImage(BufferedImage img) {
        JFrame frame = new JFrame("Image");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new JPanel() {
            private static final long serialVersionUID = 1L;

            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2d = (Graphics2D) g;
                g2d.clearRect(0, 0, getWidth(), getHeight());
                g2d.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2d.scale(2, 2);
                g2d.drawImage(img, 0, 0, this);
            }
        };
        panel.setPreferredSize(new Dimension(width * 2, height * 2));
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }
}