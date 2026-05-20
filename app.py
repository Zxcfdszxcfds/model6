import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------- 配置 ----------------------
st.set_page_config(page_title="计算机视觉三大任务演示", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 页面标题 ----------------------
st.title("🖼️ 计算机视觉三大任务演示平台")
st.markdown("这是一个简化版演示，包含FCN语义分割、目标检测、实例分割的基础逻辑展示")

# ---------------------- 上传图片 ----------------------
uploaded_file = st.file_uploader("上传一张图片（jpg/png）", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="原图", use_column_width=True)

    # ---------------------- 1. 简化版FCN语义分割演示 ----------------------
    st.header("1. 简化版语义分割演示")
    if st.button("运行语义分割（演示）", key="seg_demo"):
        with st.spinner("生成分割结果..."):
            # 用简单的颜色阈值模拟分割效果
            gray = np.mean(image_np, axis=2)
            seg = np.where(gray > 128, 1, 0)
            seg_color = np.zeros_like(image_np)
            seg_color[seg == 1] = [0, 128, 0]
            seg_color[seg == 0] = [0, 0, 128]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image_np)
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(seg_color)
            axes[1].set_title("简化语义分割结果")
            axes[1].axis("off")
            st.pyplot(fig)

    # ---------------------- 2. 简化版目标检测演示 ----------------------
    st.header("2. 简化版目标检测演示")
    if st.button("运行目标检测（演示）", key="det_demo"):
        with st.spinner("生成检测结果..."):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image_np)
            # 模拟检测框（示例）
            h, w = image_np.shape[:2]
            rect = plt.Rectangle((w*0.2, h*0.2), w*0.6, h*0.6, 
                                 fill=False, color="red", linewidth=3)
            ax.add_patch(rect)
            ax.text(w*0.2, h*0.15, "Demo Object: 0.95", color="red", fontsize=12)
            ax.set_title("简化目标检测结果")
            ax.axis("off")
            st.pyplot(fig)

    # ---------------------- 3. 简化版实例分割演示 ----------------------
    st.header("3. 简化版实例分割演示")
    if st.button("运行实例分割（演示）", key="mask_demo"):
        with st.spinner("生成分割结果..."):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image_np)
            # 模拟实例掩码（圆形示例）
            h, w = image_np.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h*0.5, w*0.5
            radius = min(h, w)*0.3
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            # 绘制掩码
            masked_img = image_np.copy()
            masked_img[mask] = masked_img[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            ax.imshow(masked_img)
            ax.set_title("简化实例分割结果")
            ax.axis("off")
            st.pyplot(fig)

    # ---------------------- 4. 方法对比 ----------------------
    st.header("4. 三大任务对比说明")
    st.markdown("""
    | 任务类型 | 核心目标 | 输出形式 | 典型应用 |
    |----------|----------|----------|----------|
    | 语义分割 | 给每个像素分配类别 | 类别掩码 | 自动驾驶、医学影像 |
    | 目标检测 | 定位并识别物体位置和类别 | 边界框+类别 | 监控安防、商品识别 |
    | 实例分割 | 区分每个物体个体的像素级掩码 | 边界框+类别+掩码 | 机器人抓取、视频编辑 |
    """)

st.markdown("---")
st.caption("模式识别与图像处理 - A6 作业演示平台 | Streamlit简化版")
