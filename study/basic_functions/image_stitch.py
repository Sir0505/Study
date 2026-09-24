import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_black_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


image1 = cv2.imread("study/source/left_01.png")
image2 = cv2.imread("study/source/right_01.png")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# --- SIFT 特征匹配 ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5),
    dict(checks=50)
)
matches = flann.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]

points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# --- 计算公共画布 ---
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]

corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
corners2 = cv2.perspectiveTransform(
    np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2), H)

all_corners = np.concatenate([corners1, corners2], axis=0)
xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)

T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
H = T @ H
size = (xmax - xmin, ymax - ymin)

# --- Warp 到公共画布 ---
warped1 = cv2.warpPerspective(image1, T, size)
warped2 = cv2.warpPerspective(image2, H, size)

mask1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), T, size) > 0
mask2 = cv2.warpPerspective(np.full((h2, h2 if False else h2), 255, np.uint8), H, size) > 0
# 上面那行小心，改成下面这样：
mask2 = cv2.warpPerspective(np.full((h2, w2), 255, np.uint8), H, size) > 0

# --- 1) 增益补偿：在重叠区把 warped2 拉到 warped1 的亮度 ---
overlap = mask1 & mask2
if overlap.sum() > 0:
    for c in range(3):
        mean1 = float(warped1[..., c][overlap].mean())
        mean2 = float(warped2[..., c][overlap].mean())
        if mean2 > 1e-6:
            gain = mean1 / mean2
            # 限制增益范围，避免颜色失真过大
            gain = np.clip(gain, 0.5, 2.0)
            warped2[..., c] = np.clip(
                warped2[..., c].astype(np.float32) * gain, 0, 255
            ).astype(np.uint8)

# --- 2) 距离变换加权融合 ---
d1 = cv2.distanceTransform((mask1 * 255).astype(np.uint8), cv2.DIST_L2, 5)
d2 = cv2.distanceTransform((mask2 * 255).astype(np.uint8), cv2.DIST_L2, 5)
d1 = d1.astype(np.float32)
d2 = d2.astype(np.float32)
w_sum = d1 + d2
w_sum[w_sum == 0] = 1.0
alpha1 = (d1 / w_sum)[..., None]
alpha2 = (d2 / w_sum)[..., None]

panorama = (warped1.astype(np.float32) * alpha1 +
            warped2.astype(np.float32) * alpha2)
panorama = np.clip(panorama, 0, 255).astype(np.uint8)

optimized_panorama = crop_black_edges(panorama)

optimized_panorama =optimized_panorama[195:570,0:1100]

cv2.imwrite("study/output/image_stich.bmp",optimized_panorama)

plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(optimized_panorama, cv2.COLOR_BGR2RGB))
plt.title('Panorama with Gain Compensation + Distance-Weighted Blending')
plt.axis('off')
plt.show()