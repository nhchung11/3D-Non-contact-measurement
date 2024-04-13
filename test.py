<<<<<<< HEAD
import my_lib
import numpy as np
import cv2
bin_path = r'D:\python\wavelet_intern\data\bin8.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()


my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
=======
import numpy as np

# Tạo một mảng numpy 2D để minh họa
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Khởi tạo danh sách để chứa các phần tử có cấu trúc (hàng, cột, giá trị)
result_list = []

# Duyệt qua từng hàng và cột
for i in range(array_2d.shape[0]):
    for j in range(array_2d.shape[1]):
        # Lấy giá trị từ mảng
        value = array_2d[i, j]
        # Thêm vào danh sách phần tử có cấu trúc
        result_list.append((i, j, value))

# In danh sách kết quả
print("Danh sách kết quả:", result_list)
>>>>>>> 2c8a231 (add files)
