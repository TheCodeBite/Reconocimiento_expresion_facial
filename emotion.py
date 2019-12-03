import cv2
import glob
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
import dlib
from sklearn.svm import SVC
from tkinter import *
from tkinter.filedialog import askopenfilename
from shutil import copyfile
from joblib import dump, load

url_img = 'C:/Users/kamh1/OneDrive/Documentos/GitHub/FINAL C3 IA/163367 MARTINEZ HERNANDEZ/dataset/test.jpg'
emotions = ["miedo", "feliz", "neutral", "tristeza"]  # lista de sentimientos
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)


def get_files(emotion):
    files = glob.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files))]
    # imagen a procesar
    prediction = ['dataset\\test.jpg']
    return training, prediction


def obtener_puntos_de_referencia(image):
    detections = detector(image, 1)
    # Buscamos los rostros de la persona con el uso de la libreria de dlib
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):  # Guarda coordenadas X e Y en dos listas
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        # Obtiene la media de ambos ejes para determinar el centro de gravedad
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        # calcula distancia entre cada punto y el punto central en ambos ejes
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]:  # Si la coordenada x del conjunto son las mismas, el ángulo es 0,  evitamos el error 'divide by 0' en la función
            anglenose = 0
        else:
            anglenose = int(
                #arco de grediente.
                math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        vector_de_puntos_de_referencia = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            vector_de_puntos_de_referencia.append(x)
            vector_de_puntos_de_referencia.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))
                             * 180/math.pi) - anglenose
            vector_de_puntos_de_referencia.append(dist)
            vector_de_puntos_de_referencia.append(anglerelative)

    if len(detections) < 1:
        vector_de_puntos_de_referencia = "error"
    return vector_de_puntos_de_referencia


def make_sets():
    datos_de_prediccions = []
    campos_de_prediccion = []
    training = []
    prediction = []
    for emotion in emotions:
        training, prediction = get_files(emotion)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            vector_de_puntos_de_referencia = obtener_puntos_de_referencia(
                clahe_image)
            if vector_de_puntos_de_referencia == "error":
                pass
            else:
                datos_de_prediccions.append(vector_de_puntos_de_referencia)
                campos_de_prediccion.append(emotions.index(emotion))

    return datos_de_prediccions, campos_de_prediccion,


def camera_acces():
    cap = cv2.VideoCapture(0)

    leido, frame = cap.read()

    if leido == True:
        cv2.imwrite("dataset/test.jpg", frame)
        print("Foto tomada correctamente")
    else:
        print("Error al acceder a la cámara")

    cap.release()

    cv2.imshow("Fotografia", cv2.imread('dataset/test.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    main()


def chooseFile():
    img_copy = askopenfilename()

    copyfile(img_copy, url_img)
    main()


probam1 = np.zeros((4, 10))
probam2 = np.zeros((1, 4))

accur_lin = []


def procesamiento_de_imagen():
    datos_de_entrenamiento = []
    training_labels = []
    training = []
    prediction = []
    for emotion in emotions:
        print("EVALUANDO EMOCION", emotion)
        training, prediction = get_files(emotion)
        # agregando datos a la lista de entrenamiento
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            vector_de_puntos_de_referencia = obtener_puntos_de_referencia(
                clahe_image)
            if vector_de_puntos_de_referencia == "error":
                pass
            else:
                # vector de imágenes a la lista de datos de entrenamiento
                datos_de_entrenamiento.append(vector_de_puntos_de_referencia)
                training_labels.append(emotions.index(emotion))
    print("validacion cruzada")
    x_train, y_test, y_train, y_test = train_test_split(datos_de_entrenamiento, training_labels, test_size=0.4, random_state=0)
    print("se valido con exito")
    return datos_de_entrenamiento, training_labels


def entrenamiento_red():
    for i in range(0, 10):
        datos_de_entrenamiento, training_labels = procesamiento_de_imagen()

        # gira el conjunto de entrenamiento en una matriz numpy para el clasificador
        npar_train = np.array(datos_de_entrenamiento)
        npar_trainlabs = np.array(training_labels)
        print("training SVM linear %s" % i)  # entrenamiento SVM
        clf.fit(npar_train, training_labels)
    dump(clf, 'modelo.joblib')
    print("ENTRENAMIENTO TERMINADO")


def main():
    global probam1, probam2
    probam1 = np.zeros((4, 10))
    probam2 = np.zeros((1, 4))

    clf = load('modelo.joblib')
    
    datos_de_prediccions, campos_de_prediccion = make_sets()

    npar_pred = np.array(datos_de_prediccions)  # <----
    # PARA AUMENTAR LA PROBABILIDAD POR LINE SE UTILIZO SCORE
    pred_lin = clf.score(npar_pred, campos_de_prediccion)  # <--
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)  # guarda la precision en una lista
    proba = clf.predict_proba(datos_de_prediccions)
    print("proba: ", proba)
    probam1[:, 0] = proba[1, :]
    probam2 = proba[1, :]+probam2
    # probam(:,i)=probam+proba
    proba = probam2/1

    # Probabilidad de las expresiones de cada una de las expresiones
    p1 = round(proba[0, 0], 2)
    p2 = round(proba[0, 1], 2)
    p3 = round(proba[0, 2], 2)
    p4 = round(proba[0, 3], 2)
    # hacemos 10 ejecuciones para aumentar precision
    print("Mean value lin svm: %.3f" % np.mean(accur_lin))

    # aqui se añade la imagen que quieres procesar pero aqui solo se carga para el resultado final
    frame = cv2.imread('dataset\\test.jpg')
    # ploteamos el resultado
    cv2.putText(frame, "Miedo: {}".format(p1), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Feliz: {:.2f}".format(p2), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Neutral: {}".format(p3), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Triste: {:.2f}".format(p4), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # mostramos la imagen
    cv2.imshow("Frame", frame)
    cv2.imwrite('resultado.jpg', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


ventana = Tk()
gama_valor = DoubleVar()
ventana.title("Inteligencia Artificial")  # nombre de la ventana
ventana.geometry("300x200")  # tamaño de la ventana

Button(ventana, text="Seleccionar Imagen",
       command=chooseFile).place(x=40, y=20)
Button(ventana, text="Tomar Foto", command=camera_acces).place(x=170, y=20)

Button(ventana, text="Entrenar red", command=entrenamiento_red).place(x=70, y=80)
ventana.mainloop()
