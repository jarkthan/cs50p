from config import *
from base import *



class InappropriateVideo(Exception):
    pass


class VideoPathInValid(Exception):
    pass

def sendline(msg):
    try:

        LINE_ACCESS_TOKEN="R9tUe1omDJoBxldSQngT0hlcUN2FSF5YSG87hdQ4b73"
        url = "https://notify-api.line.me/api/notify"
        
        file = {'imageFile':open("image.png",'rb')}
        msg = str(msg) + ': ' +  datetime.datetime.now().strftime("%H:%M:%S")
        data = {'message':msg}
        
        LINE_HEADERS = {"Authorization":"Bearer "+LINE_ACCESS_TOKEN}
        session = requests.Session()
        # r=session.post(url, headers=LINE_HEADERS, data=data)
        r=session.post(url, headers=LINE_HEADERS, files=file, data=data)
        y = json.loads(r.text)
        if (y["status"] == 400):
            r=session.post(url, headers=LINE_HEADERS, data=data)
        print(str('sendline : ' + msg + " : " + str(r.text)))
        # print(str('sendline : ' + msg + " : " + str(y["status"])))
        
        # sendline : SLEEP ALERT!: 09:31:29 : {"status":200,"message":"ok"}

        #time.sleep(2)
        #os.remove(imageFile)
        #print(str('end..'))
    except:
        pass

def Zoom(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    cv2Object = imutils.resize(
        cv2Object, width=(zoomSize * cv2Object.shape[1]))
    # center is simply half of the height & width (y/2,x/2)
    center = (int(cv2Object.shape[0]/2), int(cv2Object.shape[1]/2))
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (int(center[0]/zoomSize), int(center[1]/zoomSize))
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[cropScale[0]:(
        center[0] + cropScale[0]), cropScale[1]:(center[1] + cropScale[1])]
    return cv2Object


def main():

    try:
        # model path
        path_predictor = f'{BASEDIR}/models/shape_predictor_68_face_landmarks.dat'
        # creating object of gaze_tracking class
        gaze = gaze_tracking(path_predictor)
    except:
        print('Error:', e)

    # if you want to read video from webcame then pass cam_id and set webcam to true
    try:
        # cam_pipeline_str = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
##        320, 240
##        720, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        # height, width = 320, 240
        # size = height, width 

        size = cap.get(4), cap.get(3)
        height, width = int(cap.get(4)), int(cap.get(3))
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        K = [focal_length, 0.0, center[0],
             0.0, focal_length, center[1],
             0.0, 0.0, 1.0]

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left cornerq
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        c = 0

        NO_FACE_COUNTER = 0
        NO_FACE_THRESH = 10

        SELLP_COUNTER = 0
        DROWSINESS_COUNTER = 0
        prevTime = 0
        
        startTime = time.time()
        print('Start......')

        while True:
            try:
                curTime = time.time()
                _, frame = cap.read()
    ##            frame = cv2.resize(frame, (int(width /4 ), int(height / 4)), interpolation = cv2.INTER_AREA)
    ##            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
                frame = cv2.flip(frame, 1)
                frame = Zoom(frame, 2)
                frame = imutils.resize(frame, width=width ,height=height)
                


                if frame is None:
                    break

    ##            img = frame.copy()
                main_start = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = gaze.detector(gray, 0)
    ##            img = frame.copy()
                
                frame = cv2.resize(frame, (720, 480), interpolation = cv2.INTER_AREA)
    ##            height = 480
    ##            width = 720
                # print("3")
                if len(rects) != 0:
                    NO_FACE_COUNTER = 0

                    start = time.time()
                    landmarks = gaze.predictor(gray, rects[0])

                    head_pose = gaze.get_head_direction(landmarks)

                    pupil_left_coords, origin_left, center_left = gaze.get_pupil_coords(
                        'left', gray, landmarks, height, width)
                    pupil_right_coords, origin_right, center_right = gaze.get_pupil_coords(
                        'right', gray, landmarks, height, width)
                    pupil_loc = gaze.get_pupil_location(
                        pupil_left_coords, center_left, pupil_right_coords, center_right)

                    nose_end_point2D, image_points, rotation_vector, translation_vector = gaze.get_gaze(
                        landmarks, model_points, cam_matrix, dist_coeffs)

                    pitch, roll, yaw = gaze.get_head_orientation(
                        model_points, rotation_vector, translation_vector, cam_matrix, dist_coeffs)

                    gaze.get_eye_status(landmarks, frame, c)

                    status = gaze.gaze_ball_detection()
                    

                    frame = gaze.plot_gaze(frame, image_points , nose_end_point2D)
                    frame = gaze.mark_pupil(frame, origin_left, origin_right, pupil_left_coords, pupil_right_coords)

                    w = 200
                    h = 200
                    x = width-200
                    y = height-200
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    
                    # cv2.putText(frame, "Looking at {}-side".format(head_pose), (x+5, y+20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # cv2.line(frame, (x, y+30), (x+w, y+30), (0, 0, 0), 2)
                    # cv2.putText(frame, "HEAD:", (x+5, y+50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    # cv2.putText(frame, "Yaw: {:.2f}".format(yaw), (x+5, y+70),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(frame, "Roll: {:.2f}".format(roll), (x+5, y+90),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(frame, "Pitch: {:.2f}".format(pitch), (x+5, y+110),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # cv2.line(frame, (x, y+125), (x+w, y+125), (0, 0, 0), 2)
                    # cv2.putText(frame, "EYE and YAWN:", (x+5, y+145),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    # cv2.putText(frame, "Blink Duration: {:.2f}".format(gaze.blink_duration), (x+5, y+165),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.putText(frame, "Yawning: {}".format(gaze.yawn_status), (x+5, y+185),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # if status != None:
                    #     cv2.putText(frame, "{}".format(status), (30, 30),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print(str(gaze.yawn_status))
                    if (gaze.yawn_status == "sleep"):
                        SELLP_COUNTER+=1
                    else:
                        SELLP_COUNTER = 0
                    if (SELLP_COUNTER == 20):
                        SELLP_COUNTER = 0
                        msg = "SLEEP ALERT!"
                        print("SLEEP ALERT!")
                        cv2.imwrite("image.png", frame)
                        sendline(msg)
                    #
                    
                    if (gaze.yawn_status == "drowsy"):
                        SELLP_DROWSINESS+=1
                    else:
                        SELLP_DROWSINESS = 0
                    if (SELLP_DROWSINESS == 20):
                        SELLP_DROWSINESS = 0
                        msg = "DROWSINESS ALERT!"
                        print("DROWSINESS ALERT!")
                        cv2.imwrite("image.png", frame)
                        sendline(msg)


                else:
                    NO_FACE_COUNTER += 1
                    '''if face is not detected for NO_FACE_THRESH (which i have set to 30) frames,
                    then it means there is unusual movements by driver like head collapsing, looking upwards, or too much car shaking due to uneven road.
                '''
                    if NO_FACE_COUNTER >= NO_FACE_THRESH:
                        for i in range(NO_FACE_THRESH):
                            gaze.gaze_direction = np.append(
                                gaze.gaze_direction, 'no_face')
                            gaze.gaze_direction = np.delete(gaze.gaze_direction, 0)
                            gaze.blink_tracker = np.append(
                                gaze.blink_tracker, 'no_face')
                            gaze.blink_tracker = np.delete(gaze.blink_tracker, 0)
                        cv2.putText(frame, "HEAD COLLAPSE OR UNUSUAL HEAD MOVEMENT",
                                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                c += 1

                sec = curTime - prevTime
                prevTime = curTime
                fps = 1/(sec)
                fpscor=(frame.shape[1]-110,20)
                if time.time() - startTime > 1.0:
                    startTime = time.time()
                cv2.putText(frame, "FPS: {:.2f}".format(fps), fpscor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # time.sleep(0.01)
                pass

    except InappropriateVideo:
        print('Please enter valid video file')
    except VideoPathInValid:
        print('Invalid Video Path')
    except Exception as e:
        print('Error: ', e)


if __name__ == '__main__':
    main()