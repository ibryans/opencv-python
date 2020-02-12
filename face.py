import cv2

arquivo = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(arquivo)

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

    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Video", imagem)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()