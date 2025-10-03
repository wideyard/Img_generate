import os
import json

def save_image_paths_with_labels(base_directory, output_file):
    """将图像路径和对应的标注写入文本文件，以“###”分隔"""

    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历图像目录
        for index, (root, _, files) in enumerate(os.walk(base_directory)):
            for file in files:
                if file.lower().endswith('.png'):
                    image_path = os.path.join(root, file)
                    # print(image_path)
                    f.write(f"{image_path}###111\n")  # 写入图像路径和标注
                    # f.write(f"{image_path}\n")  # 写入图像路径和标注

if __name__ == "__main__":
    base_directory = "ChartGalaxy/source"
    output_txt_file = "chart_paths.txt"
    save_image_paths_with_labels(base_directory, output_txt_file)
    # print(f"wiki路径和长度已保存到 {output_txt_file}")
