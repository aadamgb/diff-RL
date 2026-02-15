import pygame
import numpy as np
from dynamics.bicopter_dynamics import BicopterDynamics
import cv2
import os
import matplotlib.pyplot as plt

class MultiTrajectoryRenderer:
    def __init__(self, drone, width=1280, height=900, scale=100, fps=60, l=0.2, video_path=None):
        pygame.init()
        self.drone = drone
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Multi-Policy Bicopter Viewer")
        self.clock = pygame.time.Clock()

        self.agents = []   # list of dicts
        self.frame = 0
        self.running = True

        self.l = l
        self.dt = 0.01
        
        # Video writer setup
        self.video_path = video_path
        self.video_writer = None
        if video_path:
            os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    def to_screen(self, x, y):
        return (
            int(x * self.scale + self.width // 2),
            int(self.height // 2 - y * self.scale)
        )

    def add_agent(self, trajectory, target_trajectory, action, control_mode, color, name=None):
        self.agents.append({
            "traj": trajectory,
            "target": target_trajectory,
            "action" : action,
            "cm" : control_mode,
            "color": color,
            "name": name
        })
    
    
    def _save_frame(self):
        """Save current pygame surface to video file."""
        if self.video_writer is None:
            return
        
        # Get the current pygame surface
        frame_surface = pygame.surfarray.array3d(self.screen)
        # Convert from (width, height, 3) to (height, width, 3)
        frame_surface = np.transpose(frame_surface, (1, 0, 2))
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_surface.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Write frame to video
        self.video_writer.write(frame_bgr)
        
        
    def draw_arrow(self, start, end, color, width=3, head_len=8, head_width=6):
        pygame.draw.line(self.screen, color, start, end, width)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.hypot(dx, dy)

        if length < 1e-6:
            return  # too small to draw arrowhead

        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        tip = end
        left = (
            end[0] - head_len * ux + head_width * px,
            end[1] - head_len * uy + head_width * py
        )
        right = (
            end[0] - head_len * ux - head_width * px,
            end[1] - head_len * uy - head_width * py
        )

        pygame.draw.polygon(self.screen, color, [tip, left, right])

    def draw_agent(self, agent):
        traj = agent["traj"]
        target = agent["target"]
        action = agent["action"]
        cm = agent["cm"]
        color = agent["color"]

        if self.frame < 1:
            return

        # Drone trace
        HISTORY = 100
        start_d = max(0, self.frame - HISTORY)
        start_t = max(0, self.frame - HISTORY - 250)
        pts = [self.to_screen(p[0], p[1]) for p in traj[start_d:self.frame + 1].squeeze()]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

        # Target trace (lighter)
        tgt_pts = [self.to_screen(p[0], p[1]) for p in target[start_t:self.frame + 1]]
        if len(tgt_pts) >= 2:
            pygame.draw.lines(self.screen, (255, 0, 0), False, tgt_pts, 1)

        # Draw current target position
        idx = min(self.frame, len(target) - 1)
        tx, ty = target[idx]
        target_px = self.to_screen(tx, ty)
        pygame.draw.circle(self.screen, (255, 50, 50), target_px, 8)  # Red target

        # Drone body
        idx = min(self.frame, len(traj) - 1)
        x, y, vx, vy, theta, omega = traj[idx].squeeze()

        # Get T and tau from control mode
        state_idx = traj[idx]
        action_idx = action[idx]
        T1, T2 = self.drone._get_control(state_idx, action_idx, cm)

        T = T1 + T2
        tau = self.l * (T2 - T1)
        
        # Handle batched output
        if T.dim() > 0:
            T = T[0]
            tau = tau[0]
        
        u2 = 0.5 * (T - tau / self.l)
        u1 = 0.5 * (T + tau / self.l)

        dir_x = -np.sin(theta)
        dir_y =  np.cos(theta)

        THRUST_SCALE = 1.0 * self.scale
        DRAG_SCALE = 2.5 * self.scale
        MAX_THRUST = 25.0

        def draw_rotor_thrust(pos_px, thrust, color):
            # mag = np.clip(thrust / MAX_THRUST, 0.0, 1.0)
            mag = thrust / MAX_THRUST
            color = (255, 0, 0) if thrust < 0.0 else color
            end = (
                int(pos_px[0] + dir_x * mag * THRUST_SCALE),
                int(pos_px[1] - dir_y * mag * THRUST_SCALE),
            )
            self.draw_arrow(
                start=pos_px,
                end=end,
                color=color,
                width=3,
                head_len=8,
                head_width=5
            )

        drag_force = self.drone._calculate_drag(vx, vy, theta)

        pos = self.to_screen(x, y)

        # Draw drag vector
        if drag_force is not None:
            drag_mag = np.hypot(drag_force[0], drag_force[1])
            if drag_mag > 1e-6:
                drag_scale =  DRAG_SCALE  # / max(drag_mag, 1e-6)
                drag_end = (
                    int(pos[0] - drag_force[0] * drag_scale),
                    int(pos[1] + drag_force[1] * drag_scale),
                )
                self.draw_arrow(
                    start=pos,
                    end=drag_end,
                    color=(100, 200, 255),
                    width=2,
                    head_len=6,
                    head_width=4
                )

        arm = self.l * self.scale
        dx = arm * np.cos(theta)
        dy = arm * np.sin(theta)

        m1 = (int(pos[0] - dx), int(pos[1] + dy))
        m2 = (int(pos[0] + dx), int(pos[1] - dy))

        pygame.draw.line(self.screen, (255, 255, 255), m1, m2, 3)
        pygame.draw.circle(self.screen, color, m1, 4)
        pygame.draw.circle(self.screen, color, m2, 4)

        # Left rotor
        draw_rotor_thrust(m1, u1, (0, 0, 0))
        # Right rotor
        draw_rotor_thrust(m2, u2, (0, 0, 0))

        # Drawing a legend
        legend_x = 10
        legend_y = 10
        line_height = 25
        
        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(legend_x, legend_y, 200, len(self.agents) * line_height + 10))
        
        font = pygame.font.Font(None, 24)
        for i, agent in enumerate(self.agents):
            y = legend_y + 5 + i * line_height
            # Draw color line
            pygame.draw.line(self.screen, agent["color"], (legend_x + 10, y + 10), (legend_x + 30, y + 10), 3)
            # Draw agent name
            name_text = font.render(agent["name"] or "Agent", True, (0, 0, 0))
            self.screen.blit(name_text, (legend_x + 40, y))


    def run(self):
        max_len = max(len(a["traj"]) for a in self.agents)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill((50, 50, 50))

            for agent in self.agents:
                self.draw_agent(agent)

            pygame.display.flip()
            
            # Save frame to video if enabled
            self._save_frame()

            self.frame = min(self.frame + 1, max_len - 1)
            self.clock.tick(self.fps)

        # Clean up video writer
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to: {self.video_path}")

        pygame.quit()


   # ===================================================
   #                     DASHBOARD
   #                 (for post analysis)
   # ===================================================

    def plot_dashboard(self):
        """Plot dashboard ."""
        for agent_index, agent in enumerate(self.agents):
            # 1. Data preparation
            # Assuming 'action' contains sent actions and 'traj' contains states
            action = agent["action"].squeeze()
            traj = agent["traj"].squeeze()
            target = agent["target"].squeeze()
            
            # omegas = traj[:, 6:8]  # Omegas from trajectory
            actions = action       # Actions (control commands)
            time_steps = np.arange(len(traj)) * self.dt

            # 2. Figure setup (Dashboard)
            # Create 2 rows and 2 column sharing X axis for easier reading
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
            fig.suptitle(f'{agent.get("name") or f"Agent {agent_index}"}', fontsize=16)
            
            # Get and format environment parameters
            env_params = self.drone.get_env_parameters()
            params_str = ', '.join([f'{k} = {v.item():.2f}' if hasattr(v, 'item') else f'{k}: {v}' 
                                     for k, v in env_params.items()])
            
            # Extract l and k1 as separate variables
            l = env_params.get('l').item() 
            k1 = env_params.get('k1').item()

            fig.text(0.5, 0.93, f'{params_str}', ha='center', fontsize=13, 
                     transform=fig.transFigure, verticalalignment='top')


            # --- Upper plot Left: Policy Outputs ---
            # Plot all action dimensions (if there are more than 2)
            agent_name = agent.get("name", "").upper()
            
            if agent_name == "SRT":
                action_labels = ["Thrust Motor 1", "Thrust Motor 2"]
            elif agent_name == "CTBR":
                action_labels = ["Collective Thrust", "Toruque"]
            elif agent_name == "LV":
                action_labels = ["vx", "vy", "kv", "kR", "kw"]
            else:
                action_labels = [f"Action Dim {i}" for i in range(actions.shape[1])]
            
            for i in range(actions.shape[1]):
                label = action_labels[i] if i < len(action_labels) else f"Action Dim {i}"
                ax1.plot(time_steps, actions[:, i], label=label, alpha=0.8) 
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Control Value')
            ax1.set_title('Control Actions vs Time')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)


            # # --- Upper plot Right: Collective Thrust and Body rate ---
            # ax2.plot(time_steps, k1*omegas[:, 0]**2 + k1*omegas[:, 1]**2, label='Collective Thrust', color='purple', linewidth=1.5)
            # ax2.plot(time_steps, (k1*omegas[:, 1]**2 - k1*omegas[:, 0]**2) , label='Torque / arm_l' , color='pink', linewidth=1.5) # TODO: Add l
            # ax2.set_ylabel('Newtons (N)')
            # ax2.set_title('Thrust & Torque vs Time')
            # ax2.legend(loc='upper right')
            # ax2.grid(True, alpha=0.3)

            # # --- Lower plot Left: Actions (Control Signals) ---
            # ax3.plot(time_steps, omegas[:, 0], label='Omega 1', color='blue', linewidth=1.5)
            # ax3.plot(time_steps, omegas[:, 1], label='Omega 2', color='cyan', linestyle='--', linewidth=1.5)
            # ax3.set_ylabel('Speed (rad/s)')
            # ax3.set_title('Rotor Speeds vs Time')
            # ax3.legend(loc='upper right')
            # ax3.grid(True, alpha=0.3)

            # --- Lower plot Right: Actions (Control Signals) ---
            distances = np.sqrt((target[:, 0] - traj[:, 0])**2 + (target[:, 1] - traj[:, 1])**2).detach().cpu().numpy()
            ax4.plot(time_steps, distances, label='Distance' , color='red', linewidth=1.5)
            ax4.set_ylabel('Meters (m)')
            mse = np.mean(distances**2)
            ax4.axhline(y=np.sqrt(mse), color='red', linestyle='--', linewidth=1.5, label=f'√MSE = {np.sqrt(mse):.3f}')
            ax4.set_title(f'Distance to Target vs Time\n MSE = {mse:.3f}')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            # Automatic adjustment to prevent title overlap
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

        # Show all windows simultaneously
        plt.show()
