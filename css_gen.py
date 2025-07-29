import base64
import mimetypes
import random
import os
import json
import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import numpy as np
from graphviz import Source
import subprocess
from typing import List, Dict, Tuple, Union

# 配置目录
chem_dir = "/llm/datasets/OCRSets/CASIA-CSDB"
chem_dir_txt = "/llm/wyr/projects/any-text/image_paths_with_labels.txt"
background_image_dir = '/llm/datasets/unsplash/unsplash_extracted/unsplash'
wiki_dir = "/llm/wyr/projects/any-text/wiki"
wiki_dir_txt = "/llm/wyr/projects/any-text/wiki_paths_with_length.txt"
latex_image_dir = "/llm/datasets/OCRSets/im2latex/images"
latex_json = "/llm/datasets/OCRSets/im2latex/im2latex.json"
scores_dir = "/llm/wyr/projects/any-text/scores"
scores_txt = "/llm/wyr/projects/any-text/scores_paths_with_length.txt"
chart_txt = "/llm/wyr/projects/any-text/chart_paths.txt"
MAX_LENGTH = 3600  # 最大尺寸限制

# 文本布局基础参数（图像适应相关参数）
MIN_CHAR_WIDTH = 6
MAX_CHAR_WIDTH = 12
MIN_CHAR_HEIGHT = 14
MAX_CHAR_HEIGHT = 24
MIN_LINE_LENGTH = 8
MAX_LINE_LENGTH = 45
ADAPTIVE_FACTOR = 0.8  # 图像特征影响因子（0-1）

# 紧凑布局参数
MIN_SPACING = 10  # 元素之间最小间距（像素）
SORT_BY_AREA = True  # 按面积排序优先放置大元素
COMPACTNESS_FACTOR = 0.8  # 紧凑度权重（0-1，越高越紧凑）
# 新增：宽高比目标范围（让宽高更接近）
TARGET_ASPECT_RATIO_RANGE = [0.8, 1.2]  

# 字体和样式配置
GOOGLE_FONTS = [
    "Roboto", "Open Sans", "Lato", "Montserrat", "Oswald", "Raleway", "Merriweather",
    "PT Sans", "Noto Sans", "Source Sans Pro", "Poppins", "Ubuntu", "Nunito", "Playfair Display",
    "Rubik", "Quicksand", "Fira Sans", "Inconsolata", "Dancing Script", "Bebas Neue"
]

TEXT_DECORATIONS = ["none", "underline", "overline"]
DECORATION_STYLES = ["solid", "double", "dotted", "dashed", "wavy"]
TEXT_TRANSFORMS = ["none", "uppercase", "lowercase", "capitalize"]

# CSS模板
CSS_TEMPLATE = """.{class_name} {{
position: relative;
overflow: visible;
font-family: '{font_family}', sans-serif;
font-weight: {font_weight};
font-style: {font_style};
font-size: {font_size};
color: {color};
background-color: {background};
text-decoration: {text_decoration};
text-decoration-style: {text_decoration_style};
text-transform: {text_transform};
letter-spacing: {letter_spacing};
word-spacing: {word_spacing};
text-shadow: {text_shadow};
text-align: {text_align};
padding: 15px;
word-wrap: break-word;
white-space: normal;
box-sizing: border-box;
z-index: 1;
}}

.{class_name}::before {{
content: '';
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-image: {background_image};
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-blend-mode: overlay;
filter: blur({blur_amount}) contrast({contrast_amount});
z-index: -1;
}}
"""

LOCAL_FONT_DIR = "./fonts"

# 辅助函数
def gen_dot_image(dot_image_num):
    path_texts = []
    for i in range(dot_image_num):
        try:
            from dot_gen import gen_random_dot
            dot_str = gen_random_dot()
            src = Source(dot_str)
            src.format = "png"
            path = f"/llm/wyr/projects/any-text/dots/dot_preview_{i}"
            path_texts.append({"path":path+".png", "text":dot_str})
            src.render(filename=path, cleanup=True)
        except Exception as e:
            print(f"生成dot图像出错: {e}")
    return path_texts

def gen_scores_image(scores_image_num):
    path_texts = []
    for i in range(scores_image_num):
        try:
            subprocess.run(["bash", "scores.sh", str(i)], check=True, timeout=10)
            with open(f"{scores_dir}/scores_{i}.txt", "r") as f:
                scores_str = f.read()
                path_texts.append({"path":f"{scores_dir}/scores_{i}.png", "text":scores_str})
        except Exception as e:
            print(f"生成scores图像出错: {e}")
    return path_texts

def get_random_image_from_txt(txt_file, num_images):
    path_texts = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            image_paths = f.readlines()
            if len(image_paths) < num_images:
                num_images = len(image_paths)
            samples = random.sample(image_paths, num_images)
            for s in samples:
                parts = s.split("###")
                if len(parts) >= 2:
                    path_texts.append({"path":parts[0], "text":parts[1]})
    except Exception as e:
        print(f"从文本文件获取图像出错: {e}")
    return path_texts

def get_random_image_from_json(json_file, num_images):
    path_texts = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if len(json_data) < num_images:
                num_images = len(json_data)
            samples = random.sample(json_data, num_images)
            for s in samples:
                path_texts.append({"path":os.path.join(latex_image_dir, s["image_name"]), "text":s["text"]})
    except Exception as e:
        print(f"从JSON文件获取图像出错: {e}")
    return path_texts

def get_random_texts_from_txt(wiki_dir_txt, num_texts):
    text_samples = []
    try:
        with open(wiki_dir_txt, 'r', encoding='utf-8') as f:
            wiki_paths = f.readlines()
            if len(wiki_paths) < num_texts:
                num_texts = len(wiki_paths)
            samples = random.sample(wiki_paths, num_texts)
            for path in samples:
                try:
                    with open(path.strip('\n'), 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                        long_texts = [t for t in data if len(t['text'])>100]
                        if long_texts:
                            text_samples.append(random.choice(long_texts)['text'])
                except Exception as e:
                    print(f"读取文本文件出错 {path}: {e}")
    except Exception as e:
        print(f"从文本文件获取文本出错: {e}")
    return text_samples

def get_image_dimensions(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        print(f"获取图片尺寸出错 {path}: {e}")
        return (300, 200)

# 紧凑的动态规划布局算法
def gen_compact_layout(elements: List[Dict]) -> Tuple[List[Dict], Tuple[int, int]]:
    """
    修复元素重叠问题的紧凑布局算法：
    1. 增加完整的重叠检测机制
    2. 修复变量作用域问题
    3. 确保元素位置计算准确
    """
    # 1. 预处理元素（按面积降序排序）
    sorted_elements = sorted(elements, key=lambda x: -x['area'])
    dp = {
        (0, 0): {
            'area': 0,
            'elements': [],
            'last_w': 0,
            'last_h': 0
        }
    }

    # 2. 修正重叠检测函数（增加new_elem参数）
    def is_overlap(new_pos, existing_elems, new_elem):
        """
        检查新元素位置是否与现有元素重叠
        
        参数:
            new_pos: 新元素的位置 (x, y)
            existing_elems: 现有元素列表
            new_elem: 新元素对象（包含宽高信息）
        """
        nx, ny = new_pos
        # 使用传入的new_elem参数获取宽高（修复作用域问题）
        nw, nh = new_elem['width'], new_elem['height']
        
        # 检查与所有现有元素是否重叠
        for elem in existing_elems:
            ex, ey = elem['pos']
            ew, eh = elem['elem']['width'], elem['elem']['height']
            
            # 矩形重叠检测公式
            # 不重叠的四种情况：新元素在现有元素的左/右/上/下方
            if not (nx + nw < ex or       # 新元素在左侧
                    nx > ex + ew or       # 新元素在右侧
                    ny + nh < ey or       # 新元素在上方
                    ny > ey + eh):        # 新元素在下方
                return True  # 重叠
        return False  # 不重叠

    # 3. 逐个放置元素，确保不重叠
    for new_elem in sorted_elements:
        new_dp = {}
        for (w, h), state in dp.items():
            # 尝试水平放置（在右侧）
            new_w = w + new_elem['width'] + MIN_SPACING
            new_h = max(h, new_elem['height'])
            new_pos = (w + MIN_SPACING, 0)
            
            if new_w <= MAX_LENGTH and new_h <= MAX_LENGTH:
                # 调用时传入new_elem参数
                if not is_overlap(new_pos, state['elements'], new_elem):
                    key = (new_w, new_h)
                    new_area = state['area'] + new_elem['area']
                    if key not in new_dp or new_area > new_dp[key]['area']:
                        new_dp[key] = {
                            'area': new_area,
                            'elements': state['elements'] + [{'elem': new_elem, 'pos': new_pos}],
                            'last_w': new_w,
                            'last_h': new_h
                        }

            # 尝试垂直放置（在下侧）
            new_w2 = max(w, new_elem['width'])
            new_h2 = h + new_elem['height'] + MIN_SPACING
            new_pos2 = (0, h + MIN_SPACING)
            
            if new_w2 <= MAX_LENGTH and new_h2 <= MAX_LENGTH:
                if not is_overlap(new_pos2, state['elements'], new_elem):
                    key2 = (new_w2, new_h2)
                    new_area2 = state['area'] + new_elem['area']
                    if key2 not in new_dp or new_area2 > new_dp[key2]['area']:
                        new_dp[key2] = {
                            'area': new_area2,
                            'elements': state['elements'] + [{'elem': new_elem, 'pos': new_pos2}],
                            'last_w': new_w2,
                            'last_h': new_h2
                        }

            # 尝试填充右侧间隙
            for existing_elem in state['elements']:
                ew, eh = existing_elem['elem']['width'], existing_elem['elem']['height']
                ex, ey = existing_elem['pos']
                # 计算右侧间隙位置和大小
                gap_x = ex + ew + MIN_SPACING
                gap_y = ey
                gap_available_width = w - gap_x
                gap_available_height = eh
                
                if (gap_available_width >= new_elem['width'] and 
                    gap_available_height >= new_elem['height']):
                    gap_pos = (gap_x, gap_y)
                    if not is_overlap(gap_pos, state['elements'], new_elem):
                        key3 = (w, h)
                        new_area3 = state['area'] + new_elem['area']
                        if key3 not in new_dp or new_area3 > new_dp[key3]['area']:
                            new_dp[key3] = {
                                'area': new_area3,
                                'elements': state['elements'] + [{'elem': new_elem, 'pos': gap_pos}],
                                'last_w': w,
                                'last_h': h
                            }

        # 合并新状态
        for key, value in new_dp.items():
            if key not in dp or value['area'] > dp[key]['area']:
                dp[key] = value

    # # 4. 找到最优布局（最大面积利用率 + 最小总尺寸 + 宽高更接近）
    best_state = None
    min_total_size = float('inf')
    min_aspect_diff = float('inf')
    max_area = -1
    for state in dp.values():
        if state['area'] == 0:
            continue
        total_size = state['last_w'] * state['last_h']
        total_width = state['last_w']
        total_height = state['last_h']
        aspect_ratio = total_width / total_height if total_height != 0 else 0
        current_diff = abs(aspect_ratio - 1)
        # 优先选择面积大且总尺寸小的布局
        if (state['area'] > max_area) or (state['area'] == max_area and total_size < min_total_size and current_diff < min_aspect_diff):
            best_state = state
            max_area = state['area']
            min_total_size = total_size
            min_aspect_diff = current_diff

    if not best_state:
        return [], (0, 0)

    # 5. 行分组与垂直居中（基于最终布局）
    layout = []
    occupied_regions = []
    for item in best_state['elements']:
        elem = item['elem']
        x, y = item['pos']
        w, h = elem['width'], elem['height']
        
        # 解决重叠：强制换行逻辑
        while True:
            overlap = False
            for (rx, ry, rw, rh) in occupied_regions:
                if not (x + w < rx or x > rx + rw or y + h < ry or y > ry + rh):
                    overlap = True
                    x += MIN_SPACING * 2  # 更大的偏移避免重叠
                    if x + w > MAX_LENGTH:
                        x = 0
                        y += rh + MIN_SPACING  # 基于前一行高度换行
                    break
            if not overlap:
                break
        
        layout.append({
            'id': elem['id'],
            'type': elem['type'],
            'path': elem['path'],
            'text': elem['text'],
            'position': (x, y),
            'size': (w, h)
        })
        occupied_regions.append((x, y, w, h))

    # 行分组（确保行边界精准）
    rows = []
    for elem in layout:
        x, y = elem['position']
        w, h = elem['size']
        elem_top = y
        elem_bottom = y + h
        
        placed = False
        for i, row in enumerate(rows):
            # 更严格的垂直重叠判断
            if (elem_top < row['max_bottom'] and elem_bottom > row['min_top']):
                rows[i]['elements'].append(elem)
                rows[i]['min_top'] = min(row['min_top'], elem_top)
                rows[i]['max_bottom'] = max(row['max_bottom'], elem_bottom)
                placed = True
                break
        if not placed:
            rows.append({
                'elements': [elem],
                'min_top': elem_top,
                'max_bottom': elem_bottom
            })

    # 行内垂直居中调整（原有逻辑保留）
    adjusted_layout = []
    for row in rows:
        row_height = row['max_bottom'] - row['min_top']
        for elem in row['elements']:
            x, y = elem['position']
            w, h = elem['size']
            center_offset = (row_height - h) / 2
            new_y = row['min_top'] + center_offset
            elem['position'] = (x, new_y)
            adjusted_layout.append(elem)

    total_width = max([e['position'][0] + e['size'][0] for e in adjusted_layout], default=0)
    total_height = max([e['position'][1] + e['size'][1] for e in adjusted_layout], default=0)

    return adjusted_layout, (total_width, total_height), rows


def get_text_dimensions(text: str, image_features=None) -> Tuple[int, int]:
    """
    计算更紧凑的文本块尺寸，减少留白并适应整体布局
    
    参数:
        text: 文本内容
        image_features: 可选，图像特征字典，用于协调文本与图像尺寸
    """
    text_length = len(text)
    if text_length == 0:
        return (150, 80)  # 空文本默认最小尺寸
    
    # 1. 基础参数：使用更小的字符间距和更紧凑的行高
    base_char_width = 7.5  # 字符宽度（比之前更小）
    base_line_height = 1.2  # 行高倍数（更紧凑）
    min_font_size = 12
    max_font_size = 18
    
    # 2. 根据文本长度动态调整字体大小（长文本用小字体，短文本用大字体）
    if text_length < 50:
        font_size = max_font_size  # 短文本用较大字体
        chars_per_line = 15  # 每行字符少，宽度窄
    elif text_length < 200:
        font_size = int((max_font_size - min_font_size) * 0.7 + min_font_size)
        chars_per_line = 25
    else:
        font_size = min_font_size  # 长文本用较小字体
        chars_per_line = 35  # 每行字符多，宽度适中
    
    # 3. 精确计算行数和基础尺寸（减少冗余空间）
    lines = max(1, (text_length + chars_per_line - 1) // chars_per_line)
    content_width = chars_per_line * base_char_width * (font_size / 14)  # 字体大小修正
    content_height = lines * font_size * base_line_height
    
    # 4. 添加必要的内边距（最小化留白）
    padding_x = 10  # 水平内边距（比之前小）
    padding_y = 8   # 垂直内边距（比之前小）
    final_width = int(content_width + padding_x * 2)
    final_height = int(content_height + padding_y * 2)
    
    # 5. 与图像尺寸协调（如果提供了图像特征）
    if image_features:
        # 文本宽度不超过图像平均宽度的1.2倍
        if final_width > image_features.get('avg_width', 400) * 1.2:
            final_width = int(image_features['avg_width'] * 1.2)
            # 按新宽度重新计算每行字符数和行数
            adjusted_chars = int((final_width - padding_x * 2) / (base_char_width * (font_size / 14)))
            if adjusted_chars > 0:
                lines = max(1, (text_length + adjusted_chars - 1) // adjusted_chars)
                final_height = int(lines * font_size * base_line_height + padding_y * 2)
        
        # 确保文本高度不小于图像平均高度的1/3（避免过矮）
        min_height = max(final_height, int(image_features.get('avg_height', 300) * 0.3))
        final_height = min_height
    
    # 6. 设置最小尺寸限制（避免过小难以阅读）
    final_width = max(final_width, 120)  # 最小宽度
    final_height = max(final_height, 60)  # 最小高度
    
    return (final_width, final_height)


# 生成HTML布局
def generate_dynamic_layout(images_and_texts, total_width, total_height, rows):
    container_style = (
        f"position: relative; width: {total_width}px; height: {total_height}px; "
        f"margin: 0; padding: 0; box-sizing: border-box; background: #fff;"
    )
    flow_html = f'<div class="flow-container" style="{container_style}">'

    # 1. 绘制行横线（严格分割行，基于上一行最大 bottom）
    for i in range(1, len(rows)):
        prev_row = rows[i-1]
        curr_row = rows[i]
        # 修复：行线位于上一行所有元素的最大 bottom
        prev_row_max_bottom = max(
            elem['position'][1] + elem['size'][1] 
            for elem in prev_row['elements']
        )
        line_y = int(prev_row_max_bottom)  
        flow_html += f'''
            <div class="row-line" style="
                position: absolute; left: 0; right: 0; top: {line_y}px; 
                height: 2px; background: #000000; 
                z-index: 9999; 
            "></div>
        '''

    # 2. 绘制列竖线（覆盖行内所有元素的垂直范围）
    for row in rows:
        elements = row['elements']
        if len(elements) <= 1:
            continue  

        elements_sorted = sorted(elements, key=lambda e: e['position'][0])
        # 修复：取行内所有元素的最小 top 和最大 bottom
        row_min_top = min(elem['position'][1] for elem in elements)
        row_max_bottom = max(elem['position'][1] + elem['size'][1] for elem in elements)
        row_min_top = int(row_min_top)
        row_max_bottom = int(row_max_bottom)

        for j in range(1, len(elements_sorted)):
            prev_elem = elements_sorted[j-1]
            curr_elem = elements_sorted[j]
            prev_right = prev_elem['position'][0] + prev_elem['size'][0]
            curr_left = curr_elem['position'][0]
            line_x = int((prev_right + curr_left) / 2)  # 强制整数

            flow_html += f'''
                <div class="column-line" style="
                    position: absolute; 
                    top: {row_min_top}px;
                    height: {row_max_bottom - row_min_top}px;
                    left: {line_x}px; 
                    width: 2px; 
                    background: #000000; 
                    z-index: 9999; 
                "></div>
            '''

    css_styles = []
    out_put_labels = ""
    # 3. 渲染元素（强制整数像素，避免亚像素错位）
    for item in images_and_texts:
        x, y = item["position"]
        w, h = item["size"]
        x, y, w, h = map(int, (x, y, w, h))  # 关键修复！
        item_style = (
            f"position: absolute; left: {x}px; top: {y}px; "
            f"width: {w}px; height: {h}px; margin: 0; padding: 0; "
            f"box-sizing: border-box; overflow: visible; "
            f"z-index: 999;"
        )

        if item["type"] == "image":
            flow_html += f'''
                <div class="flow-item" style="{item_style}">
                    <img src="{image_to_data_uri(item['path'])}" 
                         style="width: 100%; height: 100%; object-fit: contain;">
                </div>
            '''
            out_put_labels += f"{item['text']}\n"
        else:
            class_name = f"text-style-{item['id']}"
            font_size = min(24, max(12, int(w / (len(item["text"]) // 5 + 1))))
            text_css, _ = random_css_font_style(class_name, font_size)
            text = item["text"].replace("\n", "<br>")
            flow_html += f'''
                <style>{text_css}</style>
                <div class="flow-item" style="{item_style}">
                    <div class="{class_name}" style="width: 100%; height: 100%; overflow-y: auto;">
                        {text}
                    </div>
                </div>
            '''
            css_styles.append(text_css)
            out_put_labels += f"{item['text']}\n"

    flow_html += '</div>'

    try:
        with open("/llm/wyr/projects/any-text/out_put_labels.txt", "w") as f:
            f.write(out_put_labels)
    except Exception as e:
        print(f"保存标签文件出错: {e}")

    # html_content = HTML_TEMPLATE.format(styles='\n'.join(css_styles), content=flow_html)
    # return html_content
    return flow_html

# 图片转URI（保持不变）
def image_to_data_uri(filepath):
    try:
        if os.path.exists(filepath):
            if filepath.startswith(('/llm', '/data')):
                return f"http://localhost:8000/{filepath.lstrip('/')}"
            else:
                with open(filepath, 'rb') as img_file:
                    mime_type, _ = mimetypes.guess_type(filepath)
                    mime_type = mime_type or 'application/octet-stream'
                    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                    return f"data:{mime_type};base64,{base64_data}"
        else:
            print(f"图片文件不存在: {filepath}")
    except Exception as e:
        print(f"生成图片URI出错 {filepath}: {e}")
        
# 颜色和字体相关函数
def rand_color(alpha=True):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    if alpha:
        a = round(random.uniform(0.3, 1.0), 2)
        return f"rgba({r},{g},{b},{a})"
    else:
        return f"rgb({r},{g},{b})"

def get_random_local_font():
    try:
        font_files = []
        if os.path.exists(LOCAL_FONT_DIR):
            for root, _, files in os.walk(LOCAL_FONT_DIR):
                for file in files:
                    if file.lower().endswith((".ttf", ".otf", ".ttc", ".woff", ".woff2")):
                        font_files.append(os.path.join(root, file))
        if font_files:
            font_path = random.choice(font_files)
            font_name = os.path.splitext(os.path.basename(font_path))[0].replace(" ", "_")
            return font_name, font_path
    except Exception as e:
        print(f"获取本地字体出错: {e}")
    return None, None

def random_css_font_style(class_name, font_size=None):
    local_font_name, local_font_path = get_random_local_font()
    if local_font_name and local_font_path:
        font_family = local_font_name
        font_face_css = (
            f"@font-face {{\n"
            f"  font-family: '{local_font_name}';\n"
            f"  src: url('{image_to_data_uri(local_font_path)}');\n"
            f"}}\n"
        )
    else:
        font_family = random.choice(GOOGLE_FONTS)
        font_face_css = ""
    
    background_image = "none"
    try:
        background_image_path = get_random_background_image(background_image_dir)
        if background_image_path:
            background_image = f"url('{image_to_data_uri(background_image_path)}')"
    except Exception as e:
        print(f"获取背景图片出错: {e}")

    if font_size is None:
        font_size = f"{random.randint(14, 22)}px"
    else:
        font_size = f"{font_size}px"

    css_params = {
        "class_name": class_name,
        "font_family": font_family,
        "font_weight": random.choice(["normal", "bold", "bolder", "lighter", str(random.choice([100,200,300,400,500,600,700,800,900]))]),
        "font_style": random.choice(["normal", "italic", "oblique"]),
        "font_size": font_size,
        "color": rand_color(alpha=True),
        "background": rand_color(alpha=True),
        "background_image": background_image,
        "blur_amount": f"{random.uniform(0.5, 3):.2f}px",
        "contrast_amount": f"{random.uniform(0.3, 0.7):.2f}",
        "text_decoration": random.choice(TEXT_DECORATIONS),
        "text_decoration_style": random.choice(DECORATION_STYLES),
        "text_transform": random.choice(TEXT_TRANSFORMS),
        "letter_spacing": f"{random.uniform(-1.5, 3):.2f}px",
        "word_spacing": f"{random.uniform(0, 10):.2f}px",
        "text_shadow": "none" if random.random() < 0.3 else (
            f"{random.randint(-3, 3)}px "
            f"{random.randint(-3, 3)}px "
            f"{random.uniform(0, 10):.2f}px "
            f"{rand_color(alpha=True)}"
        ),
        "text_align": random.choice(["left", "center", "right"])
    }
    
    css = CSS_TEMPLATE.format(** css_params)
    return font_face_css + css.strip(), class_name

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Adaptive Layout</title>
<style>
body {{
    margin: 0;
    padding: 0;
    background-color: #f0f0f0;
    overflow: visible;
}}
.flow-container {{
    background-color: #fff;
}}
.flow-item {{
}}
img {{
    display: block;
    margin: 0;
    padding: 0;
}}
</style>
<style>
{styles}
</style>
</head>
<body>
{content}
</body>
</html>
"""

# 截图函数
async def html_to_png(html_content, output_png="output.png", width=MAX_LENGTH, height=MAX_LENGTH):
    with open("/llm/wyr/projects/any-text/index.html", "w") as f:
        f.write(html_content)
        
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # 不固定视口，让页面自适应内容（最大MAX_LENGTH）
        await page.set_viewport_size({"width": width, "height": height})  # 初始视口
        await page.set_content(html_content)
        # 等待内容加载完成
        await page.wait_for_timeout(1000)
        # 全屏截图（自动适应内容尺寸）
        await page.screenshot(path=output_png, full_page=True)
        await browser.close()
    return output_png

# 获取背景图片
def get_random_background_image(directory):
    try:
        if os.path.exists(directory) and os.listdir(directory):
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if image_files:
                return os.path.join(directory, random.choice(image_files))
    except Exception as e:
        print(f"获取背景图片出错: {e}")
    return None

# 生成PNG入口
def generate_png_with_dynamic_layout(images_and_texts, output_file="output.png", width=MAX_LENGTH, height=MAX_LENGTH):
    try:
        html_content = generate_dynamic_layout(images_and_texts, width, height)
        print(f"生成PNG文件: {output_file}")
        png_file = asyncio.run(html_to_png(html_content, output_file, width, height))
        return png_file
    except Exception as e:
        print(f"生成PNG过程出错: {e}")
        return None

if __name__ == "__main__":
    try:
        # 获取各类图像
        chem_image_num = 1
        chem_images_texts = get_random_image_from_txt(chem_dir_txt, chem_image_num)
        print(f"化学图像数量: {len(chem_images_texts)}")

        dot_image_num = random.randint(1, 2)
        dot_images_texts = gen_dot_image(dot_image_num)
        print(f"Dot图像数量: {len(dot_images_texts)}")

        latex_image_num = random.randint(1, 2)
        latex_images = get_random_image_from_json(latex_json, latex_image_num)
        print(f"LaTeX图像数量: {len(latex_images)}")

        chart_image_num = random.randint(1, 2)
        chart_images = get_random_image_from_txt(chart_txt, chart_image_num)
        print(f"图表数量: {len(chart_images)}")

        scores_image_num = random.randint(1, 2)
        scores_images = gen_scores_image(scores_image_num)
        print(f"分数图像数量: {len(scores_images)}")

        random_images = chem_images_texts + dot_images_texts + latex_images + chart_images + scores_images
        print(f"总图像数量: {len(random_images)}")
        
        # 获取文本
        num_texts = random.randint(2, 4)
        random_texts = get_random_texts_from_txt(wiki_dir_txt, num_texts)
        print(f"文本数量: {len(random_texts)}")

        # 构建元素列表
        elements = []
        for i, img in enumerate(random_images):
            w, h = get_image_dimensions(img['path'])
            elements.append({
                'id': i,
                'type': 'image',
                'path': img['path'],
                'width': w,
                'height': h,
                'area': w * h,
                'text': img.get('text', '')
            })

        for i, text in enumerate(random_texts):
            # 这里可优化文本尺寸，让文本更紧凑
            w, h = get_text_dimensions(text)  # 建议修改文本尺寸算法
            elements.append({
                'id': i + len(random_images),
                'type': 'text',
                'path': '',
                'width': w,
                'height': h,
                'area': w * h,
                'text': text
            })

        # 生成紧凑布局（获取 rows）
        layout, (total_width, total_height), rows = gen_compact_layout(elements)
        print(f"行数: {len(rows)}")

        # 生成动态布局（传递 rows）
        output_file = os.path.abspath("/llm/wyr/projects/any-text/sample_output.png")
        html_content = generate_dynamic_layout(layout, total_width, total_height, rows)
        png_file = asyncio.run(html_to_png(html_content, output_file, total_width, total_height))
        print("save png file to ", output_file)
    except Exception as e:
        print(f"主程序出错: {e}")
