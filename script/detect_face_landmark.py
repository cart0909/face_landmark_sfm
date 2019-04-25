#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import cv2
import face_alignment
import numpy as np


def node():
    rospy.init_node('detect_face_landmark', anonymous=True)
    cap = cv2.VideoCapture(0)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
    pub_image = rospy.Publisher("image_raw", Image, queue_size=10)
    pub_landmark = rospy.Publisher("landmark", PointCloud2, queue_size=10)
    bridge = CvBridge()
    seq = 0

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        preds = fa.get_landmarks(frame)

        if preds and len(preds) == 1:
            show_result = np.copy(frame)
            timestamp = rospy.Time().now()

            # pub landmark
            lm_msg = PointCloud2()
            lm_msg.header.stamp = timestamp
            lm_msg.header.seq = seq
            lm_msg.height = 1
            lm_msg.width = len(preds[0])
            lm_msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                             PointField('y', 4, PointField.FLOAT32, 1)]
            lm_msg.is_bigendian = False
            lm_msg.is_dense = True
            lm_msg.point_step = 8
            lm_msg.row_step = 8 * preds[0].shape[0]
            lm_msg.data = np.asarray(preds[0], np.float32).tostring()

            pub_landmark.publish(lm_msg)

            # pub raw image
            try:
                img_msg = bridge.cv2_to_imgmsg(frame, 'bgr8')
                img_msg.header.stamp = timestamp
                img_msg.header.seq = seq
                pub_image.publish(img_msg)
            except CvBridgeError as e:
                print(e)

            for lm in preds[0]:
                cv2.circle(show_result, (lm[0], lm[1]), 2, (0, 255, 0), -1)
            cv2.imshow('result', show_result)

            ++seq

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node()
    except rospy.ROSInterruptException:
        pass
