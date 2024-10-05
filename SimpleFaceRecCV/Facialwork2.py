import cv2
import face_recognition
from simple_facerec import SimpleFacerec 

#Storing the faces from a camera
sfr = SimpleFacerec()

vasim_img = ["vasim1.png", "vasim2.png", "vasim3.jpeg"]
nayef_img = ["Kaala Nayef.jpg", "nayef1.jpeg", "nayef2.jpeg"]

sfr.load_encoding_images_with_name(vasim_img, "Vasim kinda hot")
sfr.load_encoding_images_with_name(nayef_img, "Nayef cool")

sfr.load_encoding_images("images/")

#load The Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #detect faces
    face_location, face_name = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_location, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,200,0), 2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()