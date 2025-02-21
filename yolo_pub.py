#!/usr/bin/env python
import cv2
import torch
import rospy
from models.experimental import attempt_load
from utils.general import non_max_suppression
from yolo_detection.msg import yolo_msg
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

# 加载YOLOv5模型（使用本地权重文件）
model = attempt_load("/home/youshen/catkin_ws/src/yolo_detection/scripts/yolov5/best.pt",device = 'cpu')
#model_path = "/home/youshen/catkin_ws/src/yolo_detection/scripts/yolov5/best.pt"
#model = attempt_load(str(model_path), device='cpu')
# 设置检测参数
model.conf = 0.25  # 置信度阈值
model.iou = 0.45   # NMS IoU阈值

# 初始化ROS节点
rospy.init_node("yolo_ros_pub")
pub = rospy.Publisher("yolo_msg", yolo_msg, queue_size=10)

# 打开摄像头
#cap = cv2.VideoCapture(0)
video_path = '/home/youshen/catkin_ws/src/yolo_detection/scripts/yolov5/handler2.mp4'
cap = cv2.VideoCapture(video_path)

try:
    while not rospy.is_shutdown():
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to capture video frame")
            continue
        frame = cv2.resize(frame, (640, 640))
        # 将OpenCV的numpy.ndarray转换为PyTorch的Tensor
        frame_tensor = torch.from_numpy(frame).float()  # 转换为float类型
        frame_tensor = frame_tensor.permute(2, 0, 1)   # 调整维度顺序为 [C, H, W]
        frame_tensor = frame_tensor.unsqueeze(0)       # 添加batch维度 [1, C, H, W]
        frame_tensor = frame_tensor / 255.0            # 归一化到 [0, 1]

        # YOLOv5推理
        results = model(frame_tensor)
        # 检查推理结果的类型
        if isinstance(results, tuple):
            # 如果结果是tuple，提取第一个元素（模型的原始输出）
            raw_output = results[0]
        else:
            raw_output = results

        # 应用非极大值抑制（NMS）进行后处理
        pred = non_max_suppression(raw_output, conf_thres=0.25, iou_thres=0.45)

        # 提取检测结果
        if len(pred) > 0:
            detections = pred[0].cpu().numpy()  # 提取第一个batch的结果
        else:
            detections = []

        # 发布每个检测目标
        for det in detections:
            xmin, ymin, xmax, ymax, _, _ = det
            msg = yolo_msg()
            msg.xmin = int(xmin)
            msg.ymin = int(ymin)
            msg.xmax = int(xmax)
            msg.ymax = int(ymax)
            pub.publish(msg)
            print(msg)
         
        if len(pred) > 0:
            # 手动绘制检测框
            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                label = f"Class {int(cls)}: {float(conf):.2f}"
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    rospy.loginfo("YOLOv5 node shutdown")
