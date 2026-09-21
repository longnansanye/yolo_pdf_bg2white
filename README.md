# yolo_pdf_bg2white
Use yolo to detect wechat chat images and replace the dark background
# 微信深色截图 → 白底浅色气泡工具

## 📖 项目简介

本项目提供一个完整的 **YOLO 目标检测 + OpenCV 图像处理** 流水线，用于将深色模式微信聊天截图自动转换为：

- **背景** → 白色  
- **聊天气泡 / 语音条** → 浅灰色  
- **文字** → 黑色（通过保留深色像素实现）  
- **纸张照片 / 头像** → 原样保留  

适用于打印、文档归档、论文插图等需要白底截图的场景。

---

## 🧰 技术栈

| 组件 | 用途 |
|------|------|
| Python 3.13 | 主语言 |
| PyTorch (CPU) | 深度学习框架 |
| Ultralytics YOLOv8 | 目标检测模型 |
| ONNX Runtime | 模型推理部署 |
| OpenCV | 图像处理与后处理 |
| Make Sense | 网页端数据标注 |
| Labelme | 本地标注（可选） |

---

## 📂 项目结构
yolo_pdf_bg2white/

├── README.md                    # 本文件

├── dataset.yaml                 # 数据集配置文件

├── wechat_to_white.py           # 单张图片处理脚本

├── batch_process.py             # 批量处理脚本

├── convert_labels.py            # 类别映射转换脚本（如需精简类别）

│

├── obj_train_data/              # Make Sense 导出的原始标注数据

│   ├── classes.txt

│   ├── img001.jpg / .txt

│   └── ...

│

├── YOLODataset/                 # 整理后的标准数据集

│   ├── images/

│   │   ├── train/               # 训练集图片

│   │   └── val/                 # 验证集图片

│   └── labels/

│       ├── train/               # 训练集标签

│       └── val/                 # 验证集标签

│

├── runs/                        # 训练产物（自动生成）

│   └── detect/

│       └── wechat_bubble/

│           └── exp1/

│               └── weights/

│                   ├── best.pt     # 最佳模型权重

│                   ├── last.pt     # 最后一轮权重

│                   └── best.onnx   # 导出的 ONNX 模型

│

└── output/                      # 处理后图片输出目录

纯文本
---

## 🚀 快速开始

### 1. 环境安装
bash

git clone <repo_url>

cd yolo_pdf_bg2white

创建虚拟环境
python3 -m venv venv

source venv/bin/activate

升级 pip
pip install --upgrade pip

安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

安装主依赖
pip install ultralytics opencv-python numpy pillow

安装 ONNX 推理依赖
pip install onnx onnxruntime

纯文本
> ⚠️ 如果 `onnx` / `onnxruntime` 安装失败，尝试指定版本：
> ```bash
> pip install onnx==1.17.0 onnxruntime==1.20.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 2. 准备数据

#### 2.1 收集截图

收集 50~200 张深色模式微信截图，覆盖以下场景：

- 纯文字聊天（左/右气泡）
- 图片消息（特别是纸张照片）
- 语音消息
- 表情包
- 群聊
- 个人资料页
- 文件传输

#### 2.2 标注数据

**推荐方案：网页标注（无桌面环境）**

1. 打开 https://www.makesense.ai
2. 上传截图
3. 创建以下 21 个标签（顺序固定）：
0  chat_avatar_left      对方头像

1  chat_avatar_right     自己头像

2  chat_bubble_left      对方气泡

3  chat_bubble_right     自己气泡

4  chat_timestamp        时间戳

5  chat_voice_left       对方语音条

6  chat_voice_right      自己语音条

7  chat_image_left       对方图片

8  chat_image_right      自己图片

9  chat_file_left        对方文件

10 chat_file_right       自己文件

11 chat_input_bar        输入框

12 chat_name             昵称

13 profile_avatar        个人页头像

14 profile_nickname      个人页昵称

15 profile_moments       朋友圈

16 profile_message       个性签名

17 file_item             文件项

18 phone_status_bar      手机状态栏

19 normal_photo          纸张照片（关键）

20 file_path_bar         文件路径栏

纯文本
4. 使用 **Rectangle（矩形框）** 工具标注
5. 导出格式选择 **YOLO**
6. 下载 zip 并解压到 `obj_train_data/` 目录

> ⚠️ 不要使用 Polygon 工具，本项目基于目标检测（detect）而非实例分割（segment）。

#### 2.3 整理数据集
bash

创建目录结构
mkdir -p YOLODataset/images/train YOLODataset/labels/train

mkdir -p YOLODataset/images/val   YOLODataset/labels/val

复制数据
cp obj_train_data/*.jpg  YOLODataset/images/train/ 2>/dev/null

cp obj_train_data/*.jpeg YOLODataset/images/train/ 2>/dev/null

cp obj_train_data/*.png  YOLODataset/images/train/ 2>/dev/null

cp obj_train_data/*.txt  YOLODataset/labels/train/

删除 classes.txt（不是标签文件）
rm YOLODataset/labels/train/classes.txt

划分验证集（20%）
cd YOLODataset

ls images/train/ | sed 's/.[^.]*$//' | shuf > all_files.txt

total=$(wc -l < all_files.txt)

val_count=$((total * 20 / 100))

[ "$val_count" -eq 0 ] && val_count=1

head -n $val_count all_files.txt > val_files.txt

while read f; do

for ext in jpg jpeg png; do

[ -f "images/train/f.ext" ] && mv "images/train/f.ext" "images/val/"

done

[ -f "labels/train/f.txt" ] && mv "labels/train/f.txt" "labels/val/"

done < val_files.txt

rm all_files.txt val_files.txt

cd ..

验证
echo "训练集: $(ls YOLODataset/images/train/ | wc -l) 张"

echo "验证集: $(ls YOLODataset/images/val/ | wc -l) 张"

纯文本
### 3. 训练模型
bash

确保 dataset.yaml 配置正确
cat > dataset.yaml << 'EOF'

path: /new/test/yolo_pdf_bg2white/YOLODataset

train: images/train

val: images/val

nc: 21

names:

0: chat_avatar_left

1: chat_avatar_right

2: chat_bubble_left

3: chat_bubble_right

4: chat_timestamp

5: chat_voice_left

6: chat_voice_right

7: chat_image_left

8: chat_image_right

9: chat_file_left

10: chat_file_right

11: chat_input_bar

12: chat_name

13: profile_avatar

14: profile_nickname

15: profile_moments

16: profile_message

17: file_item

18: phone_status_bar

19: normal_photo

20: file_path_bar

EOF

开始训练
yolo detect train \

model=yolov8n.pt \

data=dataset.yaml \

imgsz=640 \

epochs=150 \

batch=8 \

patience=30 \

project=wechat_bubble \

name=exp1 \

device=cpu

纯文本
训练完成后，模型保存在：
runs/detect/wechat_bubble/exp1/weights/best.pt

纯文本
### 4. 导出 ONNX 模型
bash

yolo export \

model=runs/detect/wechat_bubble/exp1/weights/best.pt \

format=onnx \

imgsz=640 \

device=cpu

纯文本
导出成功后得到：
runs/detect/wechat_bubble/exp1/weights/best.onnx

纯文本
### 5. 运行推理
bash

处理单张图片
python3 wechat_to_white.py input.jpg -o output.jpg

批量处理
python3 batch_process.py --input_dir ./screenshots/ --output_dir ./output/

纯文本
---

## ⚙️ 核心处理逻辑

处理脚本 `wechat_to_white.py` 的工作流程：

1. **加载 ONNX 模型** → 初始化推理会话
2. **预处理** → Letterbox 缩放至 640×640，归一化，BGR→RGB
3. **推理** → 运行 ONNX 模型
4. **后处理** → 解析输出，NMS 去重，还原坐标
5. **图像处理**：
   - `normal_photo`、`chat_avatar_*`、`profile_*` → 保持原样
   - `chat_bubble_*`、`chat_voice_*` → 填充浅灰色（230, 235, 240）
   - 其他区域 → 填充白色（255, 255, 255）
   - 文字区域（深色像素）→ 保留原始颜色
6. **输出** → 保存处理后的图片

---

## 📊 训练结果参考

| 指标 | 值 |
|------|-----|
| 训练数据量 | 48 张 |
| 模型 | YOLOv8n |
| 最佳 epoch | 56 |
| 训练时长（CPU） | ~10 分钟 |
| mAP50（整体） | 0.83 |
| 头像检测 mAP50 | 0.99 |
| 时间戳 mAP50 | 0.99 |
| 气泡检测 mAP50 | 0.85 |
| 状态栏 mAP50 | 0.99 |

---

## ⚠️ 常见问题与注意事项

### 环境安装

| 问题 | 原因 | 解决 |
|------|------|------|
| `onnx` 安装失败 | PyPI JSON 响应截断 | 指定版本：`onnx==1.17.0 onnxruntime==1.20.1` |
| `labelme` 启动报 xcb | 无桌面环境缺少 Qt 插件 | 改用网页标注（Make Sense） |
| PyTorch 安装过大 | 装了 CUDA 版 | 加 `--index-url https://download.pytorch.org/whl/cpu` |

### 数据标注

| 问题 | 原因 | 解决 |
|------|------|------|
| 标注格式不对 | 误选了 Polygon | 用 Rectangle 标注，YOLO 导出 |
| `classes.txt` 干扰训练 | 被当作标签文件 | 放入 `labels/` 前删除 |
| 类别顺序混乱 | 标注和配置不一致 | 严格按照文档定义的顺序 |

### 训练

| 问题 | 原因 | 解决 |
|------|------|------|
| `Segment dataset requires ...` | 用了 segment 模型 | 改用 `yolo detect train` |
| 模型文件找不到 | 路径写错 | 正确路径：`runs/detect/.../weights/best.pt` |
| 稀有类 mAP=0 | 样本太少 | 每类至少 10~30 个实例 |
| 显存不足 | batch 太大 | CPU 训练设 `batch=4` |

### 推理部署

| 问题 | 原因 | 解决 |
|------|------|------|
| 检测框偏移 | 导出 `imgsz` 与训练不一致 | 统一设为 640 |
| 输出全是白色 | 后处理未做 NMS | 添加 NMS 步骤 |
| 性能慢 | CPU 推理 | 导出 OpenVINO 格式加速 |

---

## 🔧 进阶选项

### 使用更大模型（提高精度）
bash

yolo detect train model=yolov8s.pt ...   # small

yolo detect train model=yolov8m.pt ...   # medium

纯文本
### 导出 OpenVINO（Intel CPU 加速）
bash

yolo export model=best.pt format=openvino imgsz=640 device=cpu

纯文本
### 动态输入尺寸
bash

yolo export model=best.pt format=onnx imgsz=640 dynamic=True device=cpu

