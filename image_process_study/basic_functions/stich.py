import cv2
import os

# 你的图片路径（请根据实际情况修改文件名）
image_paths = [
    "image_process_study/source/1.jpg", "image_process_study/source/2.jpg", "image_process_study/source/3.jpg", 
    "image_process_study/source/4.jpg", "image_process_study/source/5.jpg", "image_process_study/source/6.jpg","image_process_study/source/7.jpg", "image_process_study/source/8.jpg",
]

# 读取图片
images = []
for path in image_paths:
    if os.path.exists(path):
        img = cv2.imread(path)
        images.append(img)

# 创建拼接器 (PANORAMA 模式适合视角变化，SCANS 模式适合平面扫描)
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# 开始拼接
status, stitched_img = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    print("拼接成功！")
    cv2.imwrite("result_panorama.jpg", stitched_img)
else:
    print(f"拼接失败，错误代码: {status}")
    # 错误代码 1: 需要更多图片
    # 错误代码 2: 无法估计单应性矩阵（特征点匹配失败）