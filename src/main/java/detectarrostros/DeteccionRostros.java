package detectarrostros;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class DeteccionRostros {

    public static void main(String[] args) throws FrameGrabber.Exception {

        // Cargamos el modelo que detecta los rostros.
        CascadeClassifier detector = new CascadeClassifier(
                "haarcascade_frontalface_alt.xml"
        );

        // Comprobamos que se ha cargado correctamente.
        if (detector.empty()) {
            System.out.println("No se ha podido cargar el modelo XML.");
            detector.close();
            return;
        }

        // Seleccionamos la primera webcam del ordenador.
        OpenCVFrameGrabber camara = new OpenCVFrameGrabber(0);

        // Convertidor entre imágenes de JavaCV y OpenCV.
        OpenCVFrameConverter.ToMat convertidor =
                new OpenCVFrameConverter.ToMat();

        // Creamos la imagen en gris y la lista de rostros.
        Mat imagenGris = new Mat();
        RectVector rostros = new RectVector();

        // Colores en orden azul, verde y rojo.
        Scalar verde = new Scalar(0, 255, 0, 0);
        Scalar rojo = new Scalar(0, 0, 255, 0);

        // Posición donde aparecerá el aviso.
        Point posicionTexto = new Point(20, 40);

        CanvasFrame ventana = null;

        try {
            // Encendemos la cámara.
            camara.start();

            // Creamos la ventana del vídeo.
            ventana = new CanvasFrame(
                    "Alarma inteligente",
                    CanvasFrame.getDefaultGamma() / camara.getGamma()
            );

            // Al cerrar la ventana, el bucle podrá terminar.
            ventana.setDefaultCloseOperation(
                    javax.swing.JFrame.DISPOSE_ON_CLOSE
            );

            // Repetimos mientras la ventana esté abierta.
            while (ventana.isVisible()) {

                // Capturamos una imagen.
                Frame captura = camara.grab();

                if (captura == null) {
                    break;
                }

                Mat imagen = convertidor.convert(captura);

                if (imagen == null || imagen.empty()) {
                    continue;
                }

                // Convertimos la imagen a escala de grises.
                cvtColor(imagen, imagenGris, COLOR_BGR2GRAY);

                // Buscamos rostros en la imagen.
                detector.detectMultiScale(imagenGris, rostros);

                // Dibujamos un rectángulo por cada rostro.
                for (int i = 0; i < rostros.size(); i++) {
                    Rect rostro = rostros.get(i);
                    rectangle(imagen, rostro, verde);
                }

                // Si hay al menos un rostro, mostramos el aviso.
                if (rostros.size() > 0) {
                    putText(
                            imagen,
                            "INTRUSO ENCONTRADO",
                            posicionTexto,
                            FONT_HERSHEY_SIMPLEX,
                            1.0,
                            rojo,
                            2,
                            LINE_AA,
                            false
                    );
                }

                // Mostramos la imagen en la ventana.
                ventana.showImage(convertidor.convert(imagen));
            }

        } finally {
            // Liberamos la cámara aunque ocurra un error.
            try {
                camara.close();
            } finally {
                if (ventana != null) {
                    ventana.dispose();
                }

                rostros.close();
                imagenGris.close();
                detector.close();
                verde.close();
                rojo.close();
                posicionTexto.close();
                convertidor.close();
            }
        }
    }
}