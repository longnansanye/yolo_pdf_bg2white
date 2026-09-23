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

├── pdf_bg_to_white.py           # PDF/单张图片转换脚本

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

训练和导出模型时安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

安装推理依赖
pip install onnxruntime opencv-python numpy pillow PyMuPDF

训练和导出模型时额外安装
pip install ultralytics onnx

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

[ -f "images/train/${f}.${ext}" ] && mv "images/train/${f}.${ext}" "images/val/"

done

[ -f "labels/train/${f}.txt" ] && mv "labels/train/${f}.txt" "labels/val/"

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

### 5. 使用 `pdf_bg_to_white.py`

脚本默认从脚本同级目录加载 `best.onnx`。如果使用虚拟环境，请先激活环境；也可以直接使用仓库中的解释器 `./bin/python`。
运行时直接使用 ONNX Runtime，不需要安装 Ultralytics、PyTorch 或 TorchVision。

#### 5.1 转换单张图片

```bash
# 指定输出路径
python3 pdf_bg_to_white.py image input.jpg output.jpg

# 省略输出路径，默认生成 input_white.jpg
python3 pdf_bg_to_white.py image input.jpg
```

单图输出格式支持 `.jpg`、`.jpeg` 和 `.png`。模型路径不是默认位置时，使用 `--model` 指定：

```bash
python3 pdf_bg_to_white.py image input.png output.png \
  --model /path/to/best.onnx
```

#### 5.2 转换 PDF

将待处理的 PDF 命名为脚本目录下的 `input.pdf`，运行：

```bash
python3 pdf_bg_to_white.py process
```

处理结果保存为 `output.pdf`；提取的原图和处理后的图片分别保存到 `extracted_images/`、`processed_images/`。

#### 5.3 从已处理图片重建 PDF

如果已经有 `processed_images/`，可以跳过重新检测，直接重建 PDF：

```bash
python3 pdf_bg_to_white.py rebuild rebuilt.pdf
```

该命令使用 `input.pdf` 作为版式模板，并从 `processed_images/` 读取处理后的图片。

#### 5.4 常用参数

```text
--model PATH    YOLO ONNX 模型路径，默认使用脚本目录下的 best.onnx
--conf FLOAT    检测置信度阈值，默认 0.25
--iou FLOAT     NMS IoU 阈值，默认 0.45
--imgsz INT     YOLO 推理尺寸，默认 640
--no-yolo       禁用 YOLO，使用图像启发式判断
```

查看完整帮助：

```bash
python3 pdf_bg_to_white.py --help
```

#### 5.5 Windows 图形界面

Windows 下直接运行 `pdf_bg_to_white.py` 或打包后的 `wechat2white-windows.exe` 会打开图形界面，可以选择：

- 转换单张 `.jpg`、`.jpeg` 或 `.png` 图片；
- 选择并转换单个 PDF，默认生成同目录下的 `<原文件名>_white.pdf`；
- 为 PDF 选择图片输出目录，目录下会保存 `extracted_images/` 和 `processed_images/`；
- 设置 YOLO 模型路径、置信度、IoU、推理尺寸，或关闭 YOLO 改用启发式判断。

Linux 仍使用上面的命令行方式，不会打开图形界面。

---

PyInstaller 打包
source /new/test/yolo_pdf_bg2white/bin/activate
# 安装 PyInstaller
pip install pyinstaller

# 打包（脚本为 pdf_bg_to_white.py）
pyinstaller --onefile --noupx \
  --add-data "runs/detect/wechat_bubble/exp1/weights/best.onnx:." \
  --name wechat2white \
  pdf_bg_to_white.py
dist/wechat2white          # Linux 可执行文件

增量训练（保留旧知识）
# 把新标注数据合并到数据集
cp 新标注图片/*.jpg YOLODataset/images/train/
cp 新标注图片/*.txt YOLODataset/labels/train/

# 重新划分验证集（保持 80%/20% 比例）
cd YOLODataset
# 先把之前的 val 挪回 train（为了重新随机划分）
mv images/val/*.jpg images/train/ 2>/dev/null
mv labels/val/*.txt labels/train/ 2>/dev/null

# 重新随机划分
ls images/train/ | sed 's/\.[^.]*$//' | shuf > all_files.txt
total=$(wc -l < all_files.txt)
val_count=$((total * 20 / 100))
[ "$val_count" -eq 0 ] && val_count=1
head -n $val_count all_files.txt > val_files.txt
while read f; do
  for ext in jpg jpeg png; do
    [ -f "images/train/${f}.${ext}" ] && mv "images/train/${f}.${ext}" "images/val/"
  done
  [ -f "labels/train/${f}.txt" ] && mv "labels/train/${f}.txt" "labels/val/"
done < val_files.txt
rm all_files.txt val_files.txt
cd ..

echo "训练集: $(ls YOLODataset/images/train/ | wc -l) 张"
echo "验证集: $(ls YOLODataset/images/val/ | wc -l) 张"
执行增量训练
bash
# 用现有的 best.pt 继续训练，epochs 可以设少一点
yolo detect train \
  model=runs/detect/wechat_bubble/exp1/weights/best.pt \
  data=dataset.yaml \
  imgsz=640 \
  epochs=50 \          # 新数据少，50 轮就够了
  batch=8 \
  patience=20 \
  project=wechat_bubble \
  name=exp2 \          # 新实验名，避免覆盖旧的
  device=cpu \
  resume=False         # 这里是继续训练，不是恢复中断
3.4 评估新旧模型对比
bash
# 旧模型验证
yolo val model=runs/detect/wechat_bubble/exp1/weights/best.pt data=dataset.yaml

# 新模型验证
yolo val model=runs/detect/wechat_bubble/exp2/weights/best.pt data=dataset.yaml

比较两者的 mAP50，如果新模型更好，就用新的。

3.5 导出新版 ONNX + 重新打包
bash
# 导出新 ONNX
yolo export \
  model=runs/detect/wechat_bubble/exp2/weights/best.pt \
  format=onnx \
  imgsz=640 \
  device=cpu

# 重新打包linux版本（更新 ONNX 文件路径）
pyinstaller --onefile \
  --add-data "runs/detect/wechat_bubble/exp2/weights/best.onnx:." \
  --name wechat2white_v2 \
  pdf_bg_to_white.py


## ⚙️ 核心处理逻辑

处理脚本 `pdf_bg_to_white.py` 的工作流程：

1. **加载 ONNX 模型** → 初始化推理会话
2. **预处理** → Letterbox 缩放至 640×640，归一化，BGR→RGB
3. **推理** → 运行 ONNX 模型
4. **后处理** → 解析输出，NMS 去重，还原坐标
5. **图像处理**：
   - `chat_image_*`、`normal_photo`、`file_item`、`chat_avatar_*` → 作为媒体/头像保护区域，尽量保持原始像素；`normal_photo` 和 `file_item` 走文档专用分支
   - `chat_bubble_*` → 填充浅灰色，气泡中的文字再映射为深色
   - `chat_voice_*` → 只作为候选框，不能直接原样保留；通过语音波形复测后仍走气泡换色路径
   - 其他区域 → 填充白色（255, 255, 255）
   - 文字区域（深色像素）→ 保留/映射为黑色，避免聊天内容消失
6. **输出** → 保存处理后的图片

### 聊天气泡与语音条判定规则

YOLO 框不是最终的像素保护规则。特别是 `chat_voice_left/right` 容易把普通的左侧文字气泡（例如“嗯嗯”“加急”）误判为语音条；如果直接把该框加入媒体保护掩膜，深色气泡就会被原样复制到白底上。

- 普通气泡使用 YOLO 的 `chat_bubble_left/right` 框和 OpenCV 中性灰气泡复测结果合并，漏检时由形状/亮度掩膜补齐。
- `chat_voice_left/right` 先检查气泡内是否存在“三个由短到长、中心基本对齐的窄亮色波形组件”。0008 一类的真实语音条能通过该复测；只有 `chat_voice` 标签而没有波形的普通文字气泡不会进入语音媒体保护路径。
- 无论是真语音条还是被误识别的普通文字气泡，都不加入原样保留集合；通过复测/普通气泡路径统一换成浅灰气泡，并把时长、声波图标或文字映射成深色。
- `chat_timestamp` 检测框优先触发完整时间戳重绘，即使时间戳靠近顶部而被通用日期筛选排除；重绘区域会扣除 `chat_name` 框及少量边缘，避免擦除昵称。
- 原样保留只针对图片消息、普通图片、`file_item` 文档内容和头像等媒体区域；语音条、普通聊天气泡及其深色背景不属于原样保留对象。

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

推荐路线

继续用 yolov8n 做基线

数据增加到 150~300 张，每类至少 10~30 个实例

再训一次 yolov8n→ exp2

如果气泡/照片仍漏检：

先试 yolov8s

数据够 300+ 再试 yolov8m

部署时：

边缘/CPU/打包分发 → n或 s

服务器/Intel CPU 用 OpenVINO → 可以上 m
