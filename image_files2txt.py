import os

def save_image_paths_with_labels(base_directory, output_file):
    """将图像路径和对应的标注写入文本文件，以“###”分隔"""
    # 定义图像目录和标注文件
    image_dirs = {
        "test": os.path.join(base_directory, "test_images"),
        "train": os.path.join(base_directory, "train_images"),
        "val": os.path.join(base_directory, "val_images"),
    }
    
    label_files = {
        "test": os.path.join(base_directory, "test.formulas.txt"),
        "train": os.path.join(base_directory, "train.formulas.txt"),
        "val": os.path.join(base_directory, "val.formulas.txt"),
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        for key in image_dirs:
            image_dir = image_dirs[key]
            label_file = label_files[key]

            # 读取标注文件
            with open(label_file, 'r', encoding='utf-8') as lf:
                labels = lf.readlines()

            # 遍历图像目录
            for index, (root, _, files) in enumerate(os.walk(image_dir)):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        image_path = os.path.join(root, file)
                        num = int(os.path.basename(image_path).split(".")[0])
                        # 获取对应的标注
                        if index < len(labels):
                            label = labels[num].strip()  # 去除换行符
                            label = label.replace(" ", "")
                            f.write(f"{image_path}###{label}\n")  # 写入图像路径和标注

if __name__ == "__main__":
    base_directory = "CASIA-CSDB/CASIA-CSDB"  # 替换为你的基础目录
    output_txt_file = "image_paths_with_labels.txt"  # 输出的文本文件名
    save_image_paths_with_labels(base_directory, output_txt_file)
    print(f"图像路径和标注已保存到 {output_txt_file}")
