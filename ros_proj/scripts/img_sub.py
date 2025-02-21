#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RedObjectDetector:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('img_sub', anonymous=True)
        
        # 创建CvBridge对象
        self.bridge = CvBridge()
        
        # 订阅图像话题
        self.image_sub = rospy.Subscriber('camera/image_raw', Image, self.image_callback)
        
        # 发布处理后的图像话题
        self.image_pub = rospy.Publisher('camera/image_processed', Image, queue_size=10)
    
    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"转换图像失败: {e}")
            return
        
        # 处理图像：提取红色物体
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # 定义红色的HSV范围
        lower_red1 = np.array([0, 170, 120])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 170, 120])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建掩码
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作：去除噪声并填充空洞
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选轮廓并绘制矩形
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 忽略太小的轮廓
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 绘制矩形
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            # 打印坐标
            coord_text = f"Center: ({x + w // 2}, {y + h // 2})"
            cv2.putText(cv_image, coord_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 发布处理后的图像
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"发布图像失败: {e}")

if __name__ == '__main__':
    try:
        detector = RedObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

