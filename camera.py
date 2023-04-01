import cv2

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):                            #class name Video and take object here cam
    def __init__(self):
        self.video=cv2.VideoCapture(0)          #read video from webcam
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        faces = faceDetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            x1, y1 = x + w, y + h
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)  # draw the rect

            cv2.line(frame, (x, y), (x + 30, y), (255, 255, 0), 6)  # top left
            cv2.line(frame, (x, y), (x, y + 30), (255, 255, 0), 6)

            cv2.line(frame, (x1, y), (x1 - 30, y), (255, 255, 0), 6)  # top right
            cv2.line(frame, (x1, y), (x1, y + 30), (255, 255, 0), 6)

            cv2.line(frame, (x, y1), (x + 30, y1), (255, 255, 0), 6)  # bottom left
            cv2.line(frame, (x, y1), (x, y1 - 30), (255, 255, 0), 6)

            cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 255, 0), 6)  # bottom right
            cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 255, 0), 6)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()