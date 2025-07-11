import numpy as np

class KF:
    def __init__(self, initial_x, initial_v, acceleration_noise: float):
        # mean of the state vector
        self.x = np.array([initial_x, initial_v])  # State vector: [position, velocity]
        self.acceleration_noise = acceleration_noise  # Noise in acceleration
        self.P = np.eye(2) * 1e-2  # Initial covariance matrix as we have 2 dimensions
    
    def predict(self, dt: float) -> None:
        # State transition matrix
        F = np.array([[1, dt],
                      [0, 1]])
        
        # Predict the next state
        new_x = F @ self.x
        
        # Update the covariance matrix
        # self.P = F P F.T + G acceleration_noise G.T
        G = np.array([[0.5 * dt**2],
                      [dt]])
        new_P = F @ self.P @ F.T + G @ G.T * self.acceleration_noise
        self.P = new_P
        self.x = new_x
    
    def update(self, measurement_value: float, measurement_variance: float) -> None:
        # y=z-Hx
        # S=H P H.T + R
        # K=P H.T S^-1
        # x=x+Ky
        # P=(I-KH)P
        z = np.array([measurement_value])
        R = np.array([[measurement_variance]])
        H = np.array([[1, 0]]).reshape(1, 2)  # Measurement matrix
        y = z - H @ self.x
        S = H @ self.P @ H.T + R  # Residual covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        new_x = self.x + K @ y
        I = np.eye(self.P.shape[0])  # Identity matrix
        new_P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        self.x = new_x
        self.P = new_P

    @property
    def pos(self) -> float:
        return self.x[0]
    
    @property
    def vel(self) -> float:
        return self.x[1]
    
    @property
    def cov(self) -> np.ndarray:
        return self.P
    
    @property
    def state(self) -> np.ndarray:
        return self.x