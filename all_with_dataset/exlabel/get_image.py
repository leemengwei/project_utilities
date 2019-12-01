import cv2 


def get_image():
    cap = cv2.VideoCapture(0)

    cap.set( cv2.CAP_PROP_FRAME_WIDTH,800)
 
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT,600)


    while True:
        ret, frame = cap.read()
        print(frame.shape)
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

class camera(object):
    def __init__( self ):
        self.cap = cv2.VideoCapture(0)
        self.cap.set( cv2.CAP_PROP_FRAME_WIDTH,1920)    
        self.cap.set( cv2.CAP_PROP_FRAME_HEIGHT,1080)

    def close(self):
        self.cap.release()

    def get_image(self):
        for i in range(5):            
            ret, frame = self.cap.read()
        return frame 

if __name__ == "__main__":
        
    c = camera()
    frm = c.get_image()

    cv2.imshow("capture", frm)
    cv2.waitKey(0)
    c.close()