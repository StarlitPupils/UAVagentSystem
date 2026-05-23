# E:/UAVagent/training/visdrone_to_yolo.py (UAVagent 1.3)
"""VisDrone2019 标注格式 → YOLOv8 训练格式转换器
VisDrone格式: <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
YOLO格式:   <class_id> <x_center> <y_center> <width> <height>  (归一化到0-1)
"""
import os
import cv2
import shutil
from pathlib import Path
from typing import Tuple, List
import random


class VisDroneToYOLO:
    """VisDrone → YOLO 数据集转换器"""
    
    # VisDrone 类别映射到 YOLO (0-indexed)
    # VisDrone: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 
    #            5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    # YOLO:    0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van, 
    #           5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
    VISDRONE_TO_YOLO = {
        1: 0,   # pedestrian
        2: 1,   # people
        3: 2,   # bicycle
        4: 3,   # car
        5: 4,   # van
        6: 5,   # truck
        7: 6,   # tricycle
        8: 7,   # awning-tricycle
        9: 8,   # bus
        10: 9,  # motor
    }
    IGNORED_CLASSES = {0, 11}  # 忽略类别
    
    NUM_CLASSES = 10
    CLASS_NAMES = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    def __init__(self, visdrone_root: str, output_dir: str, 
                 train_ratio: float = 0.8, img_size: int = 640):
        self.visdrone_root = Path(visdrone_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.img_size = img_size
        
        # 创建YOLO目录结构
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def convert_annotation(self, ann_line: str, img_w: int, img_h: int) -> Tuple[str, bool]:
        """
        转换单行VisDrone标注为YOLO格式
        Returns: (yolo_line, is_valid) - yolo格式字符串和是否有效
        """
        parts = ann_line.strip().split(',')
        if len(parts) < 8:
            return "", False
        
        frame_id = int(parts[0])
        bbox_left = float(parts[2])
        bbox_top = float(parts[3])
        bbox_width = float(parts[4])
        bbox_height = float(parts[5])
        cls_id = int(parts[7]) if len(parts) > 7 else 0
        
        # 跳过忽略类别
        if cls_id in self.IGNORED_CLASSES:
            return "", False
        
        # 映射类别
        yolo_cls = self.VISDRONE_TO_YOLO.get(cls_id)
        if yolo_cls is None:
            return "", False
        
        # 转为YOLO归一化格式: cx, cy, w, h
        cx = (bbox_left + bbox_width / 2) / img_w
        cy = (bbox_top + bbox_height / 2) / img_h
        w = bbox_width / img_w
        h = bbox_height / img_h
        
        # 裁剪到 [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        # 过滤无效框 (太小)
        if w * img_w < 5 or h * img_h < 5:
            return "", False
        
        return f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}", True
    
    def convert_sequence(self, seq_name: str, is_train: bool) -> dict:
        """
        转换单个VisDrone序列
        Returns: {"converted": int, "skipped": int, "images": int}
        """
        seq_dir = self.visdrone_root / "sequences" / seq_name
        ann_file = self.visdrone_root / "annotations" / f"{seq_name}.txt"
        
        if not seq_dir.exists():
            print(f"  ⚠️ 序列目录不存在: {seq_dir}")
            return {"converted": 0, "skipped": 0, "images": 0}
        if not ann_file.exists():
            # 尝试另一种标注路径
            ann_file = self.visdrone_root / "annotations" / seq_name / "gt.txt"
        if not ann_file.exists():
            print(f"  ⚠️ 标注文件不存在: {ann_file}")
            return {"converted": 0, "skipped": 0, "images": 0}
        
        # 读取所有标注，按帧分组
        annotations_by_frame = {}
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                if frame_id not in annotations_by_frame:
                    annotations_by_frame[frame_id] = []
                annotations_by_frame[frame_id].append(line.strip())
        
        # 获取图像列表
        image_files = sorted([f for f in os.listdir(seq_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        split = 'train' if is_train else 'val'
        converted_total = 0
        skipped_total = 0
        images_used = 0
        
        for img_file in image_files:
            # 提取帧号 (VisDrone命名格式: 0000001.jpg)
            try:
                frame_id = int(''.join(c for c in img_file if c.isdigit()))
            except ValueError:
                frame_id = int(img_file.split('.')[0])
            
            if frame_id not in annotations_by_frame:
                # 无标注帧也复制图像（YOLO训练允许空标签）
                src_img = seq_dir / img_file
                dst_img = self.output_dir / 'images' / split / f"{seq_name}_{img_file}"
                shutil.copy2(src_img, dst_img)
                # 创建空标签文件
                dst_label = self.output_dir / 'labels' / split / f"{seq_name}_{os.path.splitext(img_file)[0]}.txt"
                dst_label.touch()
                continue
            
            # 读取图像尺寸
            src_img = seq_dir / img_file
            img = cv2.imread(str(src_img))
            if img is None:
                skipped_total += 1
                continue
            img_h, img_w = img.shape[:2]
            
            # 转换所有标注
            yolo_lines = []
            for ann_line in annotations_by_frame[frame_id]:
                yolo_line, valid = self.convert_annotation(ann_line, img_w, img_h)
                if valid:
                    yolo_lines.append(yolo_line)
                    converted_total += 1
                else:
                    skipped_total += 1
            
            # 复制图像
            dst_img = self.output_dir / 'images' / split / f"{seq_name}_{img_file}"
            shutil.copy2(src_img, dst_img)
            
            # 写入YOLO标签
            dst_label = self.output_dir / 'labels' / split / f"{seq_name}_{os.path.splitext(img_file)[0]}.txt"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n' if yolo_lines else '')
            
            images_used += 1
        
        return {"converted": converted_total, "skipped": skipped_total, "images": images_used}
    
    def convert_all(self, train_sequences: List[str] = None, 
                    val_sequences: List[str] = None) -> str:
        """
        转换整个VisDrone数据集
        Returns: data.yaml 文件路径
        """
        # 获取所有序列
        seq_dir = self.visdrone_root / "sequences"
        all_sequences = sorted([d for d in os.listdir(seq_dir) 
                               if os.path.isdir(seq_dir / d)])
        
        if train_sequences is None:
            # 随机划分训练/验证集
            random.seed(42)
            random.shuffle(all_sequences)
            n_train = int(len(all_sequences) * self.train_ratio)
            train_sequences = all_sequences[:n_train]
            val_sequences = all_sequences[n_train:]
        
        print(f"VisDrone → YOLO 格式转换")
        print(f"  源数据: {self.visdrone_root}")
        print(f"  目标: {self.output_dir}")
        print(f"  训练序列: {len(train_sequences)} | 验证序列: {len(val_sequences)}")
        print(f"  类别数: {self.NUM_CLASSES}")
        
        total_stats = {"converted": 0, "skipped": 0, "images": 0}
        
        # 转换训练集
        print(f"\n📁 转换训练集...")
        for seq in train_sequences:
            stats = self.convert_sequence(seq, is_train=True)
            total_stats["converted"] += stats["converted"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["images"] += stats["images"]
            print(f"  {seq}: {stats['images']} images, {stats['converted']} boxes")
        
        # 转换验证集
        print(f"\n📁 转换验证集...")
        for seq in val_sequences:
            stats = self.convert_sequence(seq, is_train=False)
            total_stats["converted"] += stats["converted"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["images"] += stats["images"]
            print(f"  {seq}: {stats['images']} images, {stats['converted']} boxes")
        
        # 生成 data.yaml
        yaml_path = self.output_dir / "data.yaml"
        yaml_content = f"""# VisDrone → YOLO Training Dataset
path: {self.output_dir}
train: images/train
val: images/val

# Number of classes
nc: {self.NUM_CLASSES}

# Class names
names: {self.CLASS_NAMES}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n{'='*60}")
        print(f"✅ 转换完成!")
        print(f"   总图像: {total_stats['images']}")
        print(f"   总标注框: {total_stats['converted']}")
        print(f"   跳过: {total_stats['skipped']}")
        print(f"   配置文件: {yaml_path}")
        print(f"{'='*60}")
        
        return str(yaml_path)


if __name__ == "__main__":
    # 默认转换 VisDrone2019-MOT-val 中的序列
    converter = VisDroneToYOLO(
        visdrone_root="E:/datasets/VisDrone/VisDrone2019-MOT-val",
        output_dir="E:/datasets/VisDrone_YOLO",
        train_ratio=0.8,
    )
    
    # 使用前16个序列训练，其余验证
    seq_dir = Path("E:/datasets/VisDrone/VisDrone2019-MOT-val/sequences")
    all_seqs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(seq_dir / d)])
    
    n_train = max(1, int(len(all_seqs) * 0.8))
    train_seqs = all_seqs[:n_train]
    val_seqs = all_seqs[n_train:]
    
    yaml_path = converter.convert_all(train_seqs, val_seqs)
    print(f"\n👉 下一步: python training/finetune_yolo11.py --data {yaml_path}")
