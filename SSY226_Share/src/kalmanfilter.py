from numpy import np
class ExtendedKalmanFilter:
    def __init__(self, H, Q, R, x0, P0):
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, A, B, u):
        self.x = np.dot(A, self.x) + np.dot(B, u)  # State prediction using AX + BU
        self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q  # Covariance prediction


    def update(self, z):
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))