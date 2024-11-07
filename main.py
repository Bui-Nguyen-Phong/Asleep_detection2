import cv2
import numpy as np
import dlib 
import math
import paho.mqtt.client as mqtt
import time 
size_frame = 700

#cau hinh MQTT broker
'''broker_address = "192.168.1.110"
broker_port = 1883
topic = "test-topic"

client = mqtt.Client("TrangThai")
try:
    client.connect(broker_address,broker_port)
    print("ket noi thanh cong")
except Exception as e:
    print(f"ket noi that bai : {e}")
    exit(1)'''
def d (x1,y1,x2,y2):
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return d

def distance(a,b,landmark):
    x1= landmark.part(a).x
    y1= landmark.part(a).y
    x2= landmark.part(b).x
    y2= landmark.part(b).y
    return d(x1,y1,x2,y2)

font =cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA

cap = cv2.VideoCapture(0)
#data train
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

while (True):
    
    ret, frame = cap.read()
    anh = cv2.resize(frame,(size_frame,size_frame))  
    img = cv2.cvtColor(anh , cv2.COLOR_BGR2GRAY) 

    #detect face
    face_detect = face_classifier.detectMultiScale(img,scaleFactor=1.1,minNeighbors=10,minSize=(20,20)) 
    if len(face_detect) == 0:
        print ("No face detected")

    else:
        #landmark 68 points 
        for(x, y ,w , h) in face_detect:
            dlib_rect = dlib.rectangle(int (x), int(y), int (x + w), int(y + h))
            fh = h
            cv2.rectangle(anh, (x, y), (x +w, y + h), (0, 255,0), 1)
            landmark = predictor(img, dlib_rect)
            for i in range (landmark.num_parts):
                    lx = landmark.part(i).x
                    ly = landmark.part(i).y  
                    cv2.circle(anh, (lx,ly), 2, (255,0,0),-1)
        #tinh toan 

        eye_left = distance (36,39,landmark) #tinh khoang cahcs giua duoi mat
        eye_right= distance(42,45,landmark)
        
        d1 = distance(37,41,landmark) #tinh khoang cach giua 2 mi mat
        d2 = distance(38,40,landmark)
        d3 = distance(43,47,landmark)
        d4 = distance(44,46,landmark)

        ratio_1 = ((d1+d2)/2) / eye_left
        ratio_2 = ((d3+d4)/2) / eye_right

        eyes_avg_ratio = (ratio_1 + ratio_2) /2

        #print(f"{scale1:.5f} {scale2:.5f} {scale3:.5f} {scale4:.5f}" )
        eye_ratio_threshold = 0.25
        #print(f"{d1:.5f} {d2:.5f} {d3:.5f} {d4:.5f}" )
        print(eyes_avg_ratio)
        if eyes_avg_ratio <= eye_ratio_threshold:
            cv2.putText(anh,"Warning!",(10,400),font,1,(0,0,255),2,line_type)
            #client.publish(topic, "Nham mat qua 2s");
            print ("Nham mat");
        else:
            cv2.putText(anh,"OK",(10,400),font,1,(0,255,0),2,line_type)
            print ("Mo mat")
            #client.publish(topic,"mo mat")
    cv2.imshow('anh',anh)
    if cv2.waitKey(1) == ord('q'):
        #client.disconnect()
        break
cap.release()
cv2.destroyAllWindows()