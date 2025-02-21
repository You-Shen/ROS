#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def img_pub():
    # 初始化ROS节点
    rospy.init_node('img_pub', anonymous=True)
    
    # 创建Publisher，发布图像话题
    pub = rospy.Publisher('camera/image_raw', Image, queue_size=10)
    
    # 创建CvBridge对象
    bridge = CvBridge()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头
    if not cap.isOpened():
        rospy.logerr("无法打开摄像头")
        return
    
    rate = rospy.Rate(10)  # 发布频率为10Hz
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("无法读取摄像头图像")
            break
        
        # 将OpenCV图像转换为ROS图像消息
        ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
        
        # 发布图像
        pub.publish(ros_image)
        
        # 显示图像（可选）
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        rate.sleep()
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        img_pub()
    except rospy.ROSInterruptException:
        pass
