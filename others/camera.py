import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture("rtsp://username:passport@ip:port/Streaming/Channels/1")
cap = cv2.VideoCapture("rtsp:///h264/ch1/main/av_stream")
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    plt.clf()
    plt.imshow(frame)
    plt.draw()
    plt.pause(0.01)
    
    #cv2.imshow("frame",frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break


cv2.destroyAllWindows()
cap.release()
