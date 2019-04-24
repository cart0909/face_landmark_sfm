#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import face_alignment
import numpy as np


def node():
    cap = cv2.VideoCapture(0)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)

    while(True):
        ret, frame = cap.read()
        preds = fa.get_landmarks(frame)

        show_result = np.copy(frame)

        if preds and len(preds) == 1:
            for lm in preds[0]:
                cv2.circle(show_result, (lm[0], lm[1]), 2, (0, 255, 0), -1)

            cv2.imshow('result', show_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node()
    except rospy.ROSInterruptException:
        pass
