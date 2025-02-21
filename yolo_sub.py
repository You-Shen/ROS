#!/usr/bin/env python
import rospy
from yolo_detection.msg import yolo_msg

def callback(data):
    rospy.loginfo("目标的坐标信息为：xmin=%d, ymin=%d, xmax=%d, ymax=%d",
                  data.xmin, data.ymin, data.xmax, data.ymax)

def yolo_ros_sub():
    rospy.init_node('yolo_ros_sub', anonymous=True)
    rospy.Subscriber('yolo_msg', yolo_msg, callback)
    rospy.spin()

if __name__ == '__main__':
    yolo_ros_sub()

