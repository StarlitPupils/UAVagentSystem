# download_models.py
import urllib.request
import os

os.makedirs(r'E:\UAVagent\models', exist_ok=True)

models = {
    'yolo11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    'yolo11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt',
}

# 国内加速：通过ghproxy代理
PROXY = 'https://ghproxy.com/'

for name, url in models.items():
    path = os.path.join(r'E:\UAVagent\models', name)
    if os.path.exists(path):
        print(f'✅ {name} 已存在，跳过')
        continue
    print(f'下载 {name}...')
    try:
        # 尝试代理
        urllib.request.urlretrieve(PROXY + url, path)
    except:
        # 直连
        urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f'✅ {name} 下载完成 ({size_mb:.1f}MB)')
