from PIL import Image
import numpy as np

# 加载图像
image = Image.open(r".\gege.jpg").convert("L")
image.show()

# 线性点运算
linear = np.array(image)
linear = (linear * 0.5).astype(np.uint8)
Image.fromarray(linear).show()

# 分段线性点运算
segmented = np.array(image)
segmented[segmented < 128] = 0
segmented[segmented >= 128] = 255
Image.fromarray(segmented).show()

# 非线性点运算
nonlinear = np.array(image)
nonlinear = np.power(nonlinear, 2) // 256
Image.fromarray(nonlinear).show()
