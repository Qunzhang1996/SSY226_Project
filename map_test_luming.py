##更换地图为Town04
#python config.py --map Town04

import time
import carla

client = carla.Client('localhost', 2000)

world = client.get_world()

for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
#清理场景中的其他车辆    

def draw_waypoints(waypoints, road_id=None, life_time=50.0):
 
  for waypoint in waypoints:
 
    if(waypoint.road_id == road_id):
          world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                   persistent_lines=True)
 
#以距离为1的间距创建waypoints                                  
waypoints = world.get_map().generate_waypoints(distance=1.0)
#life_time 为画出的辅助标志存活时间
draw_waypoints(waypoints, road_id=39, life_time=20)

#我期望的道路id是39.

#找到这条路上的所有waypoints
filtered_waypoints = []
for waypoint in waypoints:
    if(waypoint.road_id == 39):
      filtered_waypoints.append(waypoint)
      
len_waypoints = len(filtered_waypoints)
print(f"len_waypoints: {len_waypoints}")
spawn_point = filtered_waypoints[-2].transform
spawn_point.location.z += 2
#bp为blueprint制造出来的小车
bp_lib = world.get_blueprint_library()
car_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
# vehicle = world.spawn_actor(car_bp, spawn_point)
# # 假设 vehicle 是你已经创建并放置到世界中的车辆实例
# transform = vehicle.get_transform()
# location = transform.location

# # 打印车辆的坐标
# print(f"车辆坐标: x={location.x}, y={location.y}, z={location.z}")

# 确保 spawn_point 是有效的位置
# 例如：spawn_point = carla.Transform(carla.Location(x=10.0, y=20.0, z=1.0))

car = None
try:
    car = world.spawn_actor(car_bp, spawn_point)
except Exception as e:
    print(f"放置车辆时出错: {e}")

if car is not None:
    # 稍等片刻以确保车辆坐标已更新
    time.sleep(1)  # 等待1秒

    transform = car.get_transform()
    location_car = transform.location

    # 打印车辆的坐标
    print(f"car的坐标: x={location_car.x}, y={location_car.y}, z={location_car.z}")
else:
    print("车辆未成功创建")



truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
# x_truck, y_truck, z_truck = location_car.x+100,location_car.y-3.5, 1.6630632877349854
# start_point_truck = min(spawn_point, key=lambda spawn_point: spawn_point.location.distance(carla.Location(x_truck, y_truck, z_truck)))
# truck = world.try_spawn_actor(truck_bp, start_point_truck)
spawn_point.location.y += 3.5
spawn_point.location.x -= 100
truck  = world.spawn_actor(truck_bp, spawn_point)
time.sleep(1)  # 等待1秒
location_truck = truck.get_transform().location
print(f"truck的坐标: x={location_truck.x}, y={location_truck.y}, z={location_truck.z}")