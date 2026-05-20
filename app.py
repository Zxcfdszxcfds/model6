import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ---------------------- 配置 ----------------------
st.set_page_config(page_title="计算机视觉三大任务对比平台", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 数据与模型加载 ----------------------
@st.cache_resource
def load_models():
    # 1. FCN 语义分割模型
    fcn = fcn_resnet50(pretrained=True).to(device).eval()
    # 2. Faster R-CNN 目标检测模型
    faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    # 3. Mask R-CNN 实例分割模型
    mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    return fcn, faster_rcnn, mask_rcnn

fcn, faster_rcnn, mask_rcnn = load_models()

# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 语义分割颜色映射（简化版）
def get_segmentation_colors():
    return np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0]
    ], dtype=np.uint8)

colors = get_segmentation_colors()

# ---------------------- 模块1：FCN 语义分割 ----------------------
def run_fcn(image_tensor):
    with torch.no_grad():
        output = fcn(image_tensor)['out']
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        color_pred = colors[pred]
    return color_pred

# ---------------------- 模块2：Faster R-CNN 目标检测 ----------------------
def run_faster_rcnn(image_tensor, threshold=0.5):
    with torch.no_grad():
        predictions = faster_rcnn(image_tensor)
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    # 过滤低置信度检测结果
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]

def draw_detections(image, boxes, scores, labels):
    image = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# ---------------------- 模块3：Mask R-CNN 实例分割 ----------------------
def run_mask_rcnn(image_tensor, threshold=0.5):
    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy().squeeze(1)
    # 过滤低置信度结果
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep], masks[keep]

def draw_masks(image, masks):
    image = image.copy()
    for mask in masks:
        mask = (mask > 0.5).astype(np.uint8) * 255
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        image[mask > 0] = image[mask > 0] * 0.5 + color * 0.5
    return image

# ---------------------- Streamlit界面 ----------------------
st.title("🖼️ 计算机视觉三大任务对比平台")
tab1, tab2, tab3, tab4 = st.tabs([
    "1. FCN 语义分割",
    "2. Faster R-CNN 目标检测",
    "3. Mask R-CNN 实例分割",
    "4. 方法对比与性能分析"
])

# ---------------------- 通用上传组件 ----------------------
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png"], key="main_upload")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # ---------------------- 模块1：FCN 语义分割 ----------------------
    with tab1:
        st.header("FCN 语义分割")
        if st.button("运行FCN分割", key="run_fcn"):
            with st.spinner("分割中..."):
                seg_result = run_fcn(image_tensor)
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            axes[0].imshow(image_np)
            axes[0].set_title("原图")
            axes[0].axis('off')
            axes[1].imshow(seg_result)
            axes[1].set_title("FCN语义分割结果")
            axes[1].axis('off')
            st.pyplot(fig)

    # ---------------------- 模块2：Faster R-CNN 目标检测 ----------------------
    with tab2:
        st.header("Faster R-CNN 目标检测")
        threshold = st.slider("置信度阈值", 0.1, 0.9, 0.5, 0.1, key="det_thresh")
        if st.button("运行目标检测", key="run_det"):
            with st.spinner("检测中..."):
                boxes, scores, labels = run_faster_rcnn(image_tensor, threshold)
                # 绘制检测框
                det_result = draw_detections(image_np.copy(), boxes, scores, labels)
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            axes[0].imshow(image_np)
            axes[0].set_title("原图")
            axes[0].axis('off')
            axes[1].imshow(det_result)
            axes[1].set_title("Faster R-CNN检测结果")
            axes[1].axis('off')
            st.pyplot(fig)

    # ---------------------- 模块3：Mask R-CNN 实例分割 ----------------------
    with tab3:
        st.header("Mask R-CNN 实例分割")
        threshold_mask = st.slider("置信度阈值", 0.1, 0.9, 0.5, 0.1, key="mask_thresh")
        if st.button("运行实例分割", key="run_mask"):
            with st.spinner("分割中..."):
                boxes, scores, labels, masks = run_mask_rcnn(image_tensor, threshold_mask)
                mask_result = draw_masks(image_np.copy(), masks)
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            axes[0].imshow(image_np)
            axes[0].set_title("原图")
            axes[0].axis('off')
            axes[1].imshow(mask_result)
            axes[1].set_title("Mask R-CNN实例分割结果")
            axes[1].axis('off')
            st.pyplot(fig)

    # ---------------------- 模块4：方法对比 ----------------------
    with tab4:
        st.header("三大方法对比与性能分析")
        st.subheader("1. 结果对比")
        if st.button("一键运行所有方法并对比", key="run_all"):
            with st.spinner("运行中..."):
                # 同时运行三个模型
                seg_result = run_fcn(image_tensor)
                boxes, scores, labels = run_faster_rcnn(image_tensor, 0.5)
                det_result = draw_detections(image_np.copy(), boxes, scores, labels)
                boxes_mask, scores_mask, labels_mask, masks = run_mask_rcnn(image_tensor, 0.5)
                mask_result = draw_masks(image_np.copy(), masks)
            
            fig, axes = plt.subplots(1, 4, figsize=(20,5))
            axes[0].imshow(image_np)
            axes[0].set_title("原图")
            axes[0].axis('off')
            axes[1].imshow(seg_result)
            axes[1].set_title("FCN语义分割")
            axes[1].axis('off')
            axes[2].imshow(det_result)
            axes[2].set_title("Faster R-CNN目标检测")
            axes[2].axis('off')
            axes[3].imshow(mask_result)
            axes[3].set_title("Mask R-CNN实例分割")
            axes[3].axis('off')
            st.pyplot(fig)
        
        st.subheader("2. 方法特性对比")
        st.markdown("""
        | 方法 | 任务类型 | 输出结果 | 优势 | 劣势 |
        |------|----------|----------|------|------|
        | FCN | 语义分割 | 每个像素的类别 | 全局上下文信息，分割边界平滑 | 无法区分同一类别的不同个体 |
        | Faster R-CNN | 目标检测 | 物体边界框+类别 | 速度快，适合目标定位 | 无法输出像素级分割结果 |
        | Mask R-CNN | 实例分割 | 物体边界框+类别+像素级掩码 | 同时完成检测与分割，能区分个体 | 模型复杂，推理速度较慢 |
        """)

st.markdown("---")
st.caption("模式识别与图像处理 - A6 计算机视觉三大任务对比实验")
