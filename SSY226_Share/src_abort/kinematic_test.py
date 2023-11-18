import casadi as cs
import numpy as np

class VehicleKinematic:
    def __init__(self, l=2.0):
        self.l = l
        self.lr = l / 2
        self.nx = 4  # 状态变量的数量（x, y, theta, v）
        self.nu = 2  # 控制变量的数量（a, delta）

    def create_kinematic_car_model(self):
        x = cs.SX.sym('x')
        y = cs.SX.sym('y')
        theta = cs.SX.sym('theta')
        v = cs.SX.sym('v')
        a = cs.SX.sym('a')
        delta = cs.SX.sym('delta')
        states = cs.vertcat(x, y, theta, v)
        controls = cs.vertcat(a, delta)

        dx = v * cs.cos(theta)
        dy = v * cs.sin(theta)
        dtheta = v / self.lr * cs.tan(delta)
        dv = a

        state_dynamics = cs.vertcat(dx, dy, dtheta, dv)
        self.kinematic_car_model = cs.Function('kinematic_car_model', [states, controls], [state_dynamics])

    def compute_AB(self,dt_sim=1,x_op = [1,1, 30*3.14/180, 1],u_op = [0, 0]):
        self.create_kinematic_car_model()

        # Define state and control symbolic variables
        x = cs.SX.sym('x', self.nx)
        u = cs.SX.sym('u', self.nu)

        # Get the state dynamics from the kinematic car model
        state_dynamics = self.kinematic_car_model(x, u)

        # Calculate the Jacobians for linearization
        A = cs.jacobian(state_dynamics, x)
        B = cs.jacobian(state_dynamics, u)

        # Create CasADi functions for the linearized matrices
        f_A = cs.Function('A', [x, u], [A])
        f_B = cs.Function('B', [x, u], [B])

        # Evaluate the Jacobians at the operating point
        A_op = f_A(x_op, u_op)
        B_op = f_B(x_op, u_op)

        # Discretize the linearized matrices
        newA = A_op * dt_sim + np.eye(self.nx)
        newB = B_op * dt_sim

        return newA.full(), newB.full()
    
# class C_k(IntEnum):
#     X_km, Y_km, Psi, T, V_km =range(5)

vehicle = VehicleKinematic(l=10)
A, B = vehicle.compute_AB()
print(A)
print(B)
