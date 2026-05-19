# models/README.md
模型文件需单独下载（过大，Git 不追踪）
运行以下命令自动下载：
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolov8x.pt'); YOLO('yolo11n.pt')"
可选增强模型：
python -c "from ultralytics import YOLO; YOLO('yolov10n.pt'); YOLO('rtdetr-l.pt')"
