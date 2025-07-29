import random
import math
import cv2
import numpy as np
import string
from graphviz import Source

def random_graph():
    num_points = random.randint(5, 12)
    points = [
        (random.uniform(0, 1), random.uniform(0, 1))
        for _ in range(num_points)
    ]
    distance_matrix = [
        [
            math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            for j in range(num_points)
        ]
        for i in range(num_points)
    ]
    edges = set()
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            dist = distance_matrix[i][j]
            alpha = 4
            prob = math.exp(-alpha * dist)
            if random.random() < prob:
                if random.random() < 0.5:
                    edge = (i, j)
                else:
                    edge = (j, i)
                if edge[0] != edge[1]:
                    edges.add(edge)
    edge_list = list(edges)
    edge_list.sort(key=lambda x: (x[0], x[1]))
    return points, edge_list

def draw_graph(points, edge_list):
    img_size = 500
    radius = 4
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    scaled_points = [
        (int(x * img_size), int(y * img_size))
        for x, y in points
    ]

    for i, j in edge_list:
        pt1 = scaled_points[i]
        pt2 = scaled_points[j]
        line_length = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        if line_length < 5:
            continue

        cv2.line(img, pt1, pt2, (0, 0, 0), 1, cv2.LINE_AA)

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        arrow_end = (
            int(pt1[0] + dx / 3),
            int(pt1[1] + dy / 3)
        )
        arrow_length = math.sqrt((arrow_end[0] - pt1[0]) ** 2 + (arrow_end[1] - pt1[1]) ** 2)
        tip_len = 10.0 / arrow_length if arrow_length > 0 else 0.2

        cv2.arrowedLine(img, pt1, arrow_end, (0, 0, 0), 1, line_type=cv2.LINE_AA, tipLength=tip_len)

    for pt in scaled_points:
        cv2.circle(img, pt, radius, (0, 0, 255), -1, cv2.LINE_AA)

    return img

def gen_random_label():
    chars = string.ascii_letters + string.digits
    length = random.randint(1, 5)
    return ''.join(random.choices(chars, k=length))


NODE_SHAPES = [
    "box", "box3d", "circle", "cylinder", "oval", "rect", "trapezium",
    "triangle", "parallelogram", "house", "pentagon", "hexagon", "septagon",
    "octagon", "doublecircle", "invtriangle", "invtrapezium", "invhouse",
    "note", "tab", "folder", "component", "Mcircle"
    ]
FILL_COLORS = [
    "aliceblue", "antiquewhite", "aquamarine", "azure", "beige", "bisque",
    "burlywood", "coral", "cornflowerblue", "cornsilk", "darkgray",
    "darksalmon", "darkseagreen", "floralwhite", "gainsboro", "ghostwhite",
    "gray", "greenyellow", "honeydew", "hotpink", "ivory", "khaki", "lavender",
    "lavenderblush", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
    "lightgoldenrod", "lightgray", "lightgreen", "lightpink", "lightsalmon",
    "lightskyblue", "lightsteelblue", "lightyellow", "linen", "mediumpurple",
    "mintcream", "mistyrose", "moccasin", "navajowhite", "oldlace", "orchid",
    "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
    "papayawhip", "peachpuff", "pink", "plum", "powderblue", "rosybrown",
    "salmon", "sandybrown", "seashell", "silver", "skyblue", "snow", "tan",
    "thistle", "tomato", "violet", "wheat", "white", "whitesmoke"]
ARROW_SHAPES = ["vee", "diamond", "dot", "normal"]

def gen_random_dot():
    points, edge_list = random_graph()
    labels = [gen_random_label() for _ in points]
    node_names = [f"n{i:02d}" for i in range(len(points))]
    dot_lines = ['digraph G{']

    for idx, (node_name, label) in enumerate(zip(node_names, labels)):
        shape = random.choice(NODE_SHAPES)
        fillcolor = random.choice(FILL_COLORS)

        label_with_idx = f"{idx}. {label}"
        dot_lines.append(
            f'  {node_name} [label="{label_with_idx}" shape="{shape}" style="filled" fillcolor="{fillcolor}"]'
        )

    for i, j in edge_list:
        src = node_names[i]
        dst = node_names[j]
        shape = random.choice(ARROW_SHAPES)
        dot_lines.append(f'  {src} -> {dst} [arrowhead={shape}]')
    dot_lines.append('}')
    dot_str = '\n'.join(dot_lines)
    return dot_str

def gen_random_dot_with_image():
    dot_str = gen_random_dot()
    print(dot_str)
    src = Source(dot_str)
    src.format = "png"
    png_bytes = src.pipe()
    return dot_str, png_bytes

if __name__ == "__main__":
    dot_str, png_bytes = gen_random_dot_with_image()
    with open("dot_preview.png", "wb") as f:
        f.write(png_bytes)
    print("DOT string generated and image saved as dot_preview.png")
    print(dot_str)