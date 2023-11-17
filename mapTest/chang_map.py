import carla
import random
import time

# 连接到CARLA服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # 设置超时时间

# 加载'town02'地图
world = client.load_world('town06')
# 获取当前天气参数
weather = world.get_weather()

# 设置为夜晚的天气参数
# 降低太阳的高度，增加云量，调整月亮的亮度
weather.sun_altitude_angle = -90.0  # 太阳在地平线下，模拟夜晚
weather.cloudiness = 80.0  # 增加云量
weather.precipitation = 0.0  # 如果不需要降水
weather.fog_density = 0.0  # 如果不需要雾
# 更多参数如precipitation_deposits, wind_intensity等也可以被设置

# 应用新的天气设置
world.set_weather(weather)

# 以下可以继续你的其他逻辑，比如生成车辆等



# 获取车辆蓝图
blueprint_library = world.get_blueprint_library()
car_blueprints = blueprint_library.filter('vehicle.*')

# 定义生成点
spawn_points = world.get_map().get_spawn_points()
number_of_vehicles = 2  # 我们想要生成的车辆数量

# 存储车辆的列表，以便之后可以进行额外操作
vehicles = []

# 生成车辆
for i in range(number_of_vehicles):
    # 随机选择一个车辆蓝图和生成点
    blueprint = random.choice(car_blueprints)
    if i < len(spawn_points):
        spawn_point = spawn_points[i]  # 为了确保生成点的唯一性
    else:
        spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(blueprint, spawn_point)

    # 检查车辆是否成功生成
    if vehicle is not None:
        vehicles.append(vehicle)
        vehicle.set_autopilot(True)
        print(f'Vehicle {i} is spawned')  # 输出生成信息
    else:
        print(f'Vehicle {i} failed to spawn')

# 为了看到车辆的运行，让我们等待一段时间
time_to_run = 200  # 仿真运行时间(秒)
print(f'Running simulation for {time_to_run} seconds...')
time.sleep(time_to_run)

# 仿真结束后清理
print('Destroying vehicles...')
for vehicle in vehicles:
    vehicle.destroy()
print('Done.')
