
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from KF import KF
import threading
import time

class KalmanFilterGUI:
    def __init__(self):
        # Initialize parameters
        self.initial_x = 0.0
        self.initial_v = 1.0
        self.acceleration_noise = 0.1
        self.dt = 0.001
        self.measurement_variance = 0.01
        self.measurement_steps = 2
        self.num_steps = 200
        self.real_acceleration = 0.0  # New parameter for real acceleration
        
        # Initialize simulation variables
        self.reset_simulation()
        
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('Interactive Kalman Filter Simulation', fontsize=16)
        
        # Adjust subplot positions to make room for controls
        plt.subplots_adjust(bottom=0.4, hspace=0.3)
        
        # Create sliders
        self.create_sliders()
        
        # Create buttons
        self.create_buttons()
    
        
        # Initial plot
        self.update_plot()
        
        # Start simulation
        self.running = False
        self.start_simulation()
    
    def reset_simulation(self):
        """Reset the simulation with current parameters"""
        self.kf = KF(initial_x=self.initial_x, 
                    initial_v=self.initial_v, 
                    acceleration_noise=self.acceleration_noise)
        self.real_pos = self.initial_x
        self.real_vel = self.initial_v
        self.mus = []
        self.covs = []
        self.real_positions = []
        self.measurements = []
        self.measurement_times = []
        self.step = 0.0
    
    def create_sliders(self):
        """Create parameter control sliders"""
        # Define slider positions
        slider_height = 0.02
        slider_width = 0.15
        left_col = 0.1
        right_col = 0.55
        
        # Left column sliders
        ax_init_x = plt.axes([left_col, 0.32, slider_width, slider_height])
        self.slider_init_x = Slider(ax_init_x, 'Initial Pos', -5.0, 5.0, valinit=self.initial_x)
        
        ax_init_v = plt.axes([left_col, 0.28, slider_width, slider_height])
        self.slider_init_v = Slider(ax_init_v, 'Initial Vel', -2.0, 2.0, valinit=self.initial_v)
        
        ax_accel_noise = plt.axes([left_col, 0.24, slider_width, slider_height])
        self.slider_accel_noise = Slider(ax_accel_noise, 'Accel Noise', 0.01, 1.0, valinit=self.acceleration_noise)
        
        ax_real_accel = plt.axes([left_col, 0.20, slider_width, slider_height])
        self.slider_real_accel = Slider(ax_real_accel, 'Real Accel', -2.0, 2.0, valinit=self.real_acceleration)
        
        # Right column sliders
        ax_dt = plt.axes([right_col, 0.32, slider_width, slider_height])
        self.slider_dt = Slider(ax_dt, 'Time Step', 0.0001, 0.01, valinit=self.dt)
        
        ax_meas_var = plt.axes([right_col, 0.28, slider_width, slider_height])
        self.slider_meas_var = Slider(ax_meas_var, 'Meas Variance', 0.001, 0.1, valinit=self.measurement_variance)
        
        ax_meas_steps = plt.axes([right_col, 0.24, slider_width, slider_height])
        self.slider_meas_steps = Slider(ax_meas_steps, 'Meas Interval', 1, 10, valinit=self.measurement_steps, valfmt='%d')
        
        ax_num_steps = plt.axes([right_col, 0.20, slider_width, slider_height])
        self.slider_num_steps = Slider(ax_num_steps, 'Num Steps', 50, 500, valinit=self.num_steps, valfmt='%d')
        
        # Connect sliders to update function
        self.slider_init_x.on_changed(self.update_params)
        self.slider_init_v.on_changed(self.update_params)
        self.slider_accel_noise.on_changed(self.update_params)
        self.slider_real_accel.on_changed(self.update_params)
        self.slider_dt.on_changed(self.update_params)
        self.slider_meas_var.on_changed(self.update_params)
        self.slider_meas_steps.on_changed(self.update_params)
        self.slider_num_steps.on_changed(self.update_params)
    
    def create_buttons(self):
        """Create control buttons"""
        ax_reset = plt.axes([0.35, 0.15, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_clicked)
        
        ax_start_stop = plt.axes([0.5, 0.15, 0.1, 0.04])
        self.btn_start_stop = Button(ax_start_stop, 'Start/Stop')
        self.btn_start_stop.on_clicked(self.start_stop_clicked)

    
    def update_params(self, val):
        """Update parameters from sliders"""
        self.initial_x = self.slider_init_x.val
        self.initial_v = self.slider_init_v.val
        self.acceleration_noise = self.slider_accel_noise.val
        self.real_acceleration = self.slider_real_accel.val
        self.dt = self.slider_dt.val
        self.measurement_variance = self.slider_meas_var.val
        self.measurement_steps = int(self.slider_meas_steps.val)
        self.num_steps = int(self.slider_num_steps.val)
        
        # Reset simulation with new parameters
        self.reset_simulation()
        self.update_plot()
    
    def reset_clicked(self, event):
        """Reset button callback"""
        self.reset_simulation()
        self.update_plot()
    
    def start_stop_clicked(self, event):
        """Start/Stop button callback"""
        if self.running:
            self.running = False
        else:
            self.running = True
            self.start_simulation()
    
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        if not self.running:
            self.running = True
            self.sim_thread = threading.Thread(target=self.run_simulation)
            self.sim_thread.daemon = True
            self.sim_thread.start()
    
    def run_simulation(self):
        """Run the simulation loop"""
        while self.running and self.step < self.num_steps:
            # Update real system with physics
            self.real_pos = self.real_pos + self.dt * self.real_vel + 0.5 * self.real_acceleration * self.dt**2
            self.real_vel = self.real_vel + self.real_acceleration * self.dt
            
            # Store current state
            self.covs.append(self.kf.cov.copy())
            self.mus.append(self.kf.state.copy())
            self.real_positions.append(self.real_pos)
            
            # Kalman filter prediction
            self.kf.predict(self.dt)
            
            # Measurement update
            if self.step != 0 and self.step % self.measurement_steps == 0:
                measurement = self.real_pos + np.random.randn() * np.sqrt(self.measurement_variance)
                self.measurements.append(measurement)
                self.measurement_times.append(self.step)
                self.kf.update(measurement_value=measurement, 
                              measurement_variance=self.measurement_variance)
            
            self.step += 1
            
            # Update plot periodically
            if self.step % 10 == 0:
                self.update_plot()
            
            time.sleep(0.01)  # Small delay for visualization
        
        self.running = False
    
    def update_plot(self):
        """Update the plots"""
        if not self.mus:
            return
        
        # Clear previous plots
        for ax in self.axs:
            ax.clear()
        
        time_steps = np.arange(len(self.mus)) * self.dt
        
        # Position plot
        self.axs[0].set_title('Position Estimation')
        self.axs[0].plot(time_steps, [mu[0] for mu in self.mus], 'r-', label='KF Estimate', linewidth=2)
        self.axs[0].plot(time_steps, self.real_positions[:len(self.mus)], 'k--', label='True Position', linewidth=2)
        
        # Confidence bounds
        pos_std = [np.sqrt(cov[0,0]) for cov in self.covs]
        pos_estimates = [mu[0] for mu in self.mus]
        lower_bound = np.array(pos_estimates) - 2 * np.array(pos_std)
        upper_bound = np.array(pos_estimates) + 2 * np.array(pos_std)
        
        self.axs[0].fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color='green', label='95% Confidence')
        
        # Measurements
        if self.measurements:
            meas_times = np.array(self.measurement_times) * self.dt
            self.axs[0].scatter(meas_times, self.measurements, color='blue', s=30, label='Measurements', zorder=5)
        
        self.axs[0].set_ylabel('Position')
        self.axs[0].legend()
        self.axs[0].grid(True, alpha=0.3)
        
        # Velocity plot
        self.axs[1].set_title('Velocity Estimation')
        self.axs[1].plot(time_steps, [mu[1] for mu in self.mus], 'b-', label='KF Estimate', linewidth=2)
        
        # True velocity
        true_velocities = [self.initial_v + self.real_acceleration * t for t in time_steps]
        self.axs[1].plot(time_steps, true_velocities, 'k--', label='True Velocity', linewidth=2)
        
        # Confidence bounds
        vel_std = [np.sqrt(cov[1,1]) for cov in self.covs]
        vel_estimates = [mu[1] for mu in self.mus]
        lower_bound = np.array(vel_estimates) - 2 * np.array(vel_std)
        upper_bound = np.array(vel_estimates) + 2 * np.array(vel_std)
        
        self.axs[1].fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color='green', label='95% Confidence')
        
        self.axs[1].set_xlabel('Time (s)')
        self.axs[1].set_ylabel('Velocity')
        self.axs[1].legend()
        self.axs[1].grid(True, alpha=0.3)
        
        # Refresh the plot
        plt.draw()
if __name__ == "__main__":
    # Create and run the GUI
    gui = KalmanFilterGUI()
    plt.show()