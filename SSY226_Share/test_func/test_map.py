import matplotlib.pyplot as plt
import numpy as np

def generate_curve(A=10, B=0.1, x_max=50):
    x = np.linspace(0, x_max, 5000)
    y = A * np.sin(B * x)
    return x, y

# 计算每个点的方向
def calculate_direction(x, y):
    dy = np.diff(y, prepend=y[0])
    dx = np.diff(x, prepend=x[0])
    psi = np.arctan2(dy, dx)
    return psi

# 生成参考曲线
x_ref, y_ref = generate_curve()
plt.figure(figsize=(12, 6))
plt.plot(x_ref, y_ref, label="Reference Path")
plt.grid(True)
plt.show()