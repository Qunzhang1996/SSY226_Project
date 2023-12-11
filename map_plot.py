import  numpy as np
import matplotlib.pyplot as plt
lane_width=3.5

P_road_v1 = [0, 0.1, 0, 146.818-0.5 * lane_width,
             0, 0.1, 0, 146.818+0.5 * lane_width]


# Create two independent objects to represent two vehicles
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\car_state_actual.txt', 'r') as file:
    lines = file.readlines()

# 提取第一列和第二列数据
x_data = [float(line.split()[0]) for line in lines]
y_data = [float(line.split()[1]) for line in lines]       
        
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\\truck_state_actual.txt', 'r') as file:
    lines_1 = file.readlines()

# 提取第一列和第二列数据
x_data_1 = [float(line_1.split()[0]) for line_1 in lines_1]
y_data_1 = [float(line_1.split()[1]) for line_1 in lines_1]       
        
# line_v0, = plt.plot(v0_history_state[ST.S], v0_history_state[ST.N])
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_state.txt', 'r') as file:
    lines_2 = file.readlines()

# 提取第一列和第二列数据
x_data_2 = [float(line_2.split()[0]) for line_2 in lines_2]
y_data_2 = [float(line_2.split()[1]) for line_2 in lines_2]  

# line_v0, = plt.plot(v0_history_state[ST.S], v0_history_state[ST.N])
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_truck_state.txt', 'r') as file:
    lines_3 = file.readlines()

# 提取第一列和第二列数据
x_data_3 = [float(line_3.split()[0]) for line_3 in lines_3]
y_data_3 = [float(line_3.split()[1]) for line_3 in lines_3]  







# Only used for Car (CAV)
P_road_v = [lane_width, 0.1, 280, 146.818-1.5 * lane_width,
             lane_width, 0.1, 200, 146.818-0.5 * lane_width]
plt.figure(figsize=(16, 4))
plt.title("Vehicle Trajectories")
ss = np.linspace(-900, 2000, 1000)
road_right = P_road_v[0]/2 * \
    (np.tanh(P_road_v[1]*(ss-P_road_v[2]))+1)+P_road_v[3]
road_left = P_road_v[4]/2*(np.tanh(P_road_v[5]*(ss-P_road_v[6]))+1)+P_road_v[7]

# 绘制计划轨迹，并设置透明度为0.7
plt.plot(x_data_2, y_data_2, color='cyan', linestyle=':', marker='o', label='Car Planned Trajectory', alpha=0.2)
plt.plot(x_data_3, y_data_3, color='pink', linestyle=':', marker='o', label='Truck Planned Trajectory', alpha=0.2)
# 绘制实际轨迹，并设置透明度为0.7
plt.plot(x_data, y_data, color='r', linestyle='-', label='Car Actual Trajectory', alpha=1) 
plt.plot(x_data_1, y_data_1, color='g', linestyle='-', label='Truck Actual Trajectory', alpha=1)




# 添加图例

plt.plot(ss, road_right, 'k',label='Road')
plt.plot(ss, road_left, 'k')
plt.xlabel('longitudinal position')
plt.ylabel('Lateral position')

plt.legend( loc='upper right')
plt.show()

