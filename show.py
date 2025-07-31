import streamlit as st
import json
import os
from PIL import Image
import re

# 设置页面配置
st.set_page_config(
    page_title="图像标签查看器",
    page_icon="🖼️",
    layout="wide"
)

# 页面标题
st.title("图像标签查看器")

# 定义数据目录路径
data_dir = "/llm/wyr/projects/any-text/output"
json_file = os.path.join(data_dir, "output_labels.json")
images_dir = os.path.join(data_dir, "images")

# 检查文件和目录是否存在
if not os.path.exists(json_file):
    st.error(f"JSON文件 {json_file} 不存在")
    st.stop()

if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
    st.error(f"图像目录 {images_dir} 不存在")
    st.stop()

# 读取JSON数据
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
except Exception as e:
    st.error(f"读取JSON文件时出错: {str(e)}")
    st.stop()

# 过滤掉不存在的图像
valid_image_data = []
for item in image_data:
    image_name = item.get("image", "")
    if image_name:
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            valid_image_data.append(item)

if not valid_image_data:
    st.warning("没有找到有效的图像文件")
    st.stop()

# 侧边栏 - 图像选择器
with st.sidebar:
    st.header("选择图像")
    image_names = [item["image"] for item in valid_image_data]
    selected_image_name = st.selectbox(
        "选择要查看的图像:",
        image_names,
        index=0
    )

# 获取当前选中的图像数据
selected_item = next(item for item in valid_image_data if item["image"] == selected_image_name)
selected_image_path = os.path.join(images_dir, selected_image_name)
selected_labels = selected_item.get("labels", "")
# 匹配 "第X行，第X个元素：图表:" 格式的文本，并替换其中的换行符
pattern = r'第\d+行，第\d+个元素：[^:\n]+:\s*\n'
matches = re.finditer(pattern, selected_labels)

processed_text = selected_labels
for match in matches:
    original = match.group(0)
    replaced = original.replace('\n', '<br>')
    processed_text = processed_text.replace(original, replaced)

# 主内容区域
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("图像 🖼️ ")
    try:
        img = Image.open(selected_image_path)
        st.image(img, caption=selected_image_name, use_container_width=True)
    except Exception as e:
        st.error(f"无法显示图像: {str(e)}")

with col2:
    st.subheader("标签 📝 ")
    if processed_text:
        st.markdown(f"{processed_text}", unsafe_allow_html=True)
    else:
        st.info("此图像没有关联的标签")

# 显示图像总数
st.markdown(f"共有 {len(valid_image_data)} 张图像")