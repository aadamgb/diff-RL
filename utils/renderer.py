import pygame
import numpy as np
from dynamics.bicopter_dynamics import BicopterDynamics

drone = BicopterDynamics()

class MultiTrajectoryRenderer:
    def __init__(self, width=1280, height=900, scale=100, fps=60, l=0.2):
        pygame.init()
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
        pts = [self.to_screen(p[0], p[1]) for p in traj[start_d:self.frame + 1]]
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
        x, y, _, _, theta, _ = traj[idx]

        # Get T and tau from control mode
        state_idx = traj[idx]
        action_idx = action[idx]
        T1, T2 = drone._get_control(state_idx, action_idx, cm)

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

        THRUST_SCALE = 1 * self.scale
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

        pos = self.to_screen(x, y)

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

            self.frame = min(self.frame + 1, max_len - 1)
            self.clock.tick(self.fps)

        pygame.quit()
