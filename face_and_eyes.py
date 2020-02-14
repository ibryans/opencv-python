import cv2

arquivoFace = 'haarcascade_frontalface_default.xml'
arquivoOlhos = 'haarcascade_eye_tree_eyeglasses.xml'
faceCascade = cv2.CascadeClassifier(arquivoFace)
olhosCascade = cv2.CascadeClassifier(arquivoOlhos)

captura = cv2.VideoCapture(0)

while True:
    s, imagem = captura.read()
    imagem = cv2.flip(imagem, 180)

    faces = faceCascade.detectMultiScale(
        imagem,
        minNeighbors = 5,
        minSize = (30, 30),
        maxSize = (200, 200)
    )

    olhos = olhosCascade.detectMultiScale(
        imagem,
        minNeighbors = 5,
        minSize = (30, 30),
        maxSize = (200, 200)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in olhos:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Video", imagem)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()