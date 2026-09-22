#!/bin/bash
# iterate_model.sh — 一键增量训练 + 打包
# 用法：./iterate_model.sh /path/to/new_images.zip

set -e

NEW_DATA_ZIP="$1"
if [ -z "$NEW_DATA_ZIP" ]; then
    echo "用法: $0 <新标注数据.zip>"
    exit 1
fi

# 1. 解压新数据
TMP_DIR=$(mktemp -d)
unzip "$NEW_DATA_ZIP" -d "$TMP_DIR"

# 2. 合并到数据集
cp "$TMP_DIR"/*.jpg YOLODataset/images/train/ 2>/dev/null || true
cp "$TMP_DIR"/*.jpeg YOLODataset/images/train/ 2>/dev/null || true
cp "$TMP_DIR"/*.png YOLODataset/images/train/ 2>/dev/null || true
cp "$TMP_DIR"/*.txt YOLODataset/labels/train/ 2>/dev/null || true
rm -f YOLODataset/labels/train/classes.txt

# 3. 重新划分验证集
cd YOLODataset
mv images/val/* images/train/ 2>/dev/null || true
mv labels/val/* labels/train/ 2>/dev/null || true
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

# 4. 计算新的实验编号
EXP_NUM=$(ls -d runs/detect/wechat_bubble/exp* 2>/dev/null | wc -l)
EXP_NUM=$((EXP_NUM + 1))

# 5. 增量训练
yolo detect train \
  model=runs/detect/wechat_bubble/exp$((EXP_NUM - 1))/weights/best.pt \
  data=dataset.yaml \
  imgsz=640 \
  epochs=50 \
  batch=8 \
  patience=20 \
  project=wechat_bubble \
  name=exp${EXP_NUM} \
  device=cpu

# 6. 导出 ONNX
yolo export \
  model=runs/detect/wechat_bubble/exp${EXP_NUM}/weights/best.pt \
  format=onnx \
  imgsz=640 \
  device=cpu

# 7. 打包
pyinstaller --onefile \
  --add-data "runs/detect/wechat_bubble/exp${EXP_NUM}/weights/best.onnx:." \
  --name "wechat2white_v${EXP_NUM}" \
  pdf_bg_to_white.py

echo ""
echo "✅ 完成！新版本: wechat2white_v${EXP_NUM}"
echo "可执行文件: dist/wechat2white_v${EXP_NUM}"
