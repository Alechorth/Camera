"""
reference frames:

TAG:
                ^ y
                |
                |
                |tag center
                O---------> x

CAMERA:


                X--------> x
                | frame center
                |
                |
                V y

F1: Flipped (180 deg) tag frame around x axis
F2: Flipped (180 deg) camera frame around x axis

The attitude of a generic frame 2 respect to a frame 1 can obtained by calculating euler(R_21.T)

We are going to obtain the following quantities:
    > from aruco library we obtain tvec and Rct, position of the tag in camera frame and attitude of the tag
    > position of the Camera in Tag axis: -R_ct.T*tvec
    > Transformation of the camera, respect to f1 (the tag flipped frame): R_cf1 = R_ct*R_tf1 = R_cf*R_f
    > Transformation of the tag, respect to f2 (the camera flipped frame): R_tf2 = Rtc*R_cf2 = R_tc*R_f
    > R_tf1 = R_cf2 an symmetric = R_f
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import math


#--- Define Tag   - [cm]
MARKER_SIZE_ROBOT = 7
MARKER_SIZE_MAT = 10
MARKER_SIZE_CAKE  = 3
    
#--- Define text 
BASE_POSITION = 50

#--- Get the camera calibration path
calib_path  = "C:\\Users\\horth_7hg6z0l\\Documents\\Python\\how_do_drones_work\\opencv\\"
camera_matrix   = np.loadtxt(calib_path +'cameraMatrix.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path +'cameraDistortion.txt', delimiter=',')

class ArucoPos():
    def __init__(self, calib_path, camera_matrix, camera_distortion):

        self.calib_path = calib_path
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

        #--- 180 deg rotation matrix around the x axis
        self.R_flip  = np.zeros((3,3), dtype=np.float32)
        self.R_flip[0,0] = 1.0
        self.R_flip[1,1] =-1.0
        self.R_flip[2,2] =-1.0

        #--- Define the aruco dictionary
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        self.parameters  = aruco.DetectorParameters_create()
        
        
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.cornerRefinementWinSize = 5        
        
        
        #--- Capture the videocamera (this may also be a video or a picture)
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        #-- Set the camera size as the one it was calibrated with (w1280,h960)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_PLAIN

        self.comparisonList = {}


    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R):
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x_rotation = math.atan2(R[2, 1], R[2, 2])
            y_rotation = math.atan2(-R[2, 0], sy)
            z_rotation = math.atan2(R[1, 0], R[0, 0])
        else:
            x_rotation = math.atan2(-R[1, 2], R[1, 1])
            y_rotation = math.atan2(-R[2, 0], sy)
            z_rotation = 0

        return np.array([x_rotation, y_rotation, z_rotation])


    def find_marker(self):
            #-- Read the camera frame
            ret, self.frame = self.cap.read()

            #-- Convert in gray scale
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) #OpenCV stores color images in Blue, Green, Red

            #-- Find all the aruco markers in the image
            self.corners, self.ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.parameters)


    def find_marker_position(self, id, size):
        ret = aruco.estimatePoseSingleMarkers(self.corners[id], size, camera_matrix, camera_distortion)
        #-- Unpack the output, get only the first
        self.rvec, self.tvec = ret[0][0,0,:], ret[1][0,0,:]
    

    def find_marker_rotation(self):
        #-- Obtain the rotation matrix tag->camera
        R_ct    = np.matrix(cv2.Rodrigues(self.rvec)[0])
        R_tc    = R_ct.T

        #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
        self.roll_marker, self.pitch_marker, self.yaw_marker = self.rotationMatrixToEulerAngles(self.R_flip*R_tc)

        #-- Now get Position and attitude f the camera respect to the marker
        self.pos_camera = -R_tc*np.matrix(self.tvec).T

        #-- Get the attitude of the camera respect to the frame
        self.roll_camera, self.pitch_camera, self.yaw_camera = self.rotationMatrixToEulerAngles(self.R_flip*R_tc)

    
    def pushComparison(self, id, ID_tvec, value = False):
        if id in self.comparisonList.keys():
            if value:
                if not np.array_equal(self.comparisonList[id], ID_tvec):
                    self.comparisonList[id] = ID_tvec
            else:    
                return
        else:
            print(id)
            self.comparisonList[id] = ID_tvec


    def distanceBetweenMarker(self, ID1, ID2):
        if ID1 in self.comparisonList.keys() and ID2 in self.comparisonList.keys():
            x1 = self.comparisonList[ID1][0]
            x2 = self.comparisonList[ID2][0]
            y1 = self.comparisonList[ID1][1]
            y2 = self.comparisonList[ID2][1]
            return np.hypot(x1-x2, y1-y2)
        else:
            return


    def display_marker_coord(self, id, rotation = False, camera_pos = False):
        #-- Draw the detected marker and put a reference frame over it
        #-- aruco.drawDetectedMarkers(self.frame, self.corners)
        cv2.drawFrameAxes(self.frame, camera_matrix, camera_distortion, self.rvec, self.tvec, 10)#-- Print the tag position in camera frame
        str_position = f"MARKER {self.ids[id]} Position x=%4.2f  y=%4.2f  z=%4.2f"%(self.tvec[0], self.tvec[1], self.tvec[2])
        cv2.putText(self.frame, str_position, (0, (BASE_POSITION)+(id*50)), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if rotation:
            #-- Print the marker's attitude respect to camera frame
            str_attitude = f"MARKER {self.ids[id]} Attitude r=%4.2f  p=%4.2f  y=%4.2f"%(math.degrees(self.roll_marker),math.degrees(self.pitch_marker),
                                math.degrees(self.yaw_marker))
            cv2.putText(self.frame, str_attitude, (0, (BASE_POSITION + 20)+(id*50)), self.font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if camera_pos:        
            str_position = "CAMERA Position x=%4.2f  y=%4.2f  z=%4.2f"%(self.pos_camera[0], self.pos_camera[1], self.pos_camera[2])
            cv2.putText(self.frame, str_position, (0, (BASE_POSITION + 100)+(id*250)), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            #-- Get the attitude of the camera respect to the frame
            str_attitude = "CAMERA Attitude r=%4.2f  p=%4.2f  y=%4.2f"%(math.degrees(self.roll_camera),math.degrees(self.pitch_camera),
                                math.degrees(self.yaw_camera))
            cv2.putText(self.frame, str_attitude, (0, (BASE_POSITION + 150)+(id*250)), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        


    def main(self):
        while 1:
            self.find_marker()
            #interesting marker detection and memorisation
            if len(self.corners):
                for i in range(0, len(self.ids)):
                    if (self.ids[i] >= 20 and self.ids[i] <= 23):
                        self.find_marker_position(i, MARKER_SIZE_MAT)
                        self.display_marker_coord(i)
                        self.pushComparison(self.ids[i][0], self.tvec)
                    if self.ids[i] == 2 or self.ids[i] == 7:
                        self.find_marker_position(i, MARKER_SIZE_ROBOT)
                        self.display_marker_coord(i)
                        self.pushComparison(self.ids[i][0], self.tvec, value=True)
            
            print("distance between Tag 2 et Tag 7 : ",self.distanceBetweenMarker(2, 7))
            
            #--- Display the frame
            cv2.imshow('frame', self.frame)
            #--- use 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    try:
        aruco_position = ArucoPos(calib_path, camera_matrix, camera_distortion)
        aruco_position.main()

    except KeyboardInterrupt:
        print("script manually interrupted")

