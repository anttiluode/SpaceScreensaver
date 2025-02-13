import os
import sys
import threading
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from PIL import Image
from scipy.ndimage import label

########################################
# 1) Logging and Configuration
########################################
logging.basicConfig(level=logging.INFO, filename='simulation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

########################################
# 2) Adaptive Tensor Units
########################################
class AdaptiveTensorUnit(nn.Module):
    """
    Each unit interacts with the 'universe_field' in 3D, 
    sampling a local region, and uses a small MLP to adapt.
    """
    def __init__(self, dimension=128, device='cpu'):
        super().__init__()
        self.dimension = dimension
        self.device = device
        
        # Physical-like properties
        self.position = torch.randn(3, device=device) * 10.0  # Random initial pos
        self.velocity = torch.zeros(3, device=device)
        self.field_signature = torch.randn(dimension, device=device)
        
        # Adaptive network for field interaction
        self.field_network = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        ).to(device)
        
        self.memory = []
        self.stability_threshold = 0.1

    def interact_with_field(self, universe_field):
        # local_field = self.sample_local_field(universe_field)
        local_field = self.sample_local_field_3d(universe_field)
        combined = torch.cat([self.field_signature, local_field])
        response = self.field_network(combined.unsqueeze(0)).squeeze(0)
        
        stability = torch.norm(response - self.field_signature)
        
        # Compute a pseudo-force from stability gradient
        # Here we do a rough approach: gradient w.r.t. position is not trivial,
        # so let's just treat stability as a scalar potential that we want to minimize.
        # We'll do a random small offset approach:
        offset = torch.randn_like(self.position) * 0.1
        test_pos = self.position + offset
        # Evaluate stability at test_pos
        # We'll just do a difference in stability approach
        old_stab = stability
        self.position += offset
        new_field = self.sample_local_field_3d(universe_field)
        combined_test = torch.cat([self.field_signature, new_field])
        test_response = self.field_network(combined_test.unsqueeze(0)).squeeze(0)
        new_stab = torch.norm(test_response - self.field_signature)
        # If new_stab is worse, revert
        if new_stab > old_stab:
            self.position -= offset
        else:
            stability = new_stab
        
        # Inertial approach
        self.velocity = 0.95 * self.velocity
        self.position += self.velocity

        # Memory
        if stability < self.stability_threshold:
            self.memory.append({
                'position': self.position.clone(),
                'signature': self.field_signature.clone(),
                'stability': stability.item()
            })
        return stability.item()

    def sample_local_field_3d(self, universe_field):
        """Extract a local 1D slice from the 3D field around self.position."""
        # We'll clamp position to valid range
        shape = universe_field.shape
        px = torch.clamp(self.position[0].long(), 0, shape[0]-1)
        py = torch.clamp(self.position[1].long(), 0, shape[1]-1)
        pz = torch.clamp(self.position[2].long(), 0, shape[2]-1)
        
        # We'll sample a small neighborhood around (px, py, pz)
        # to get a 1D vector of size dimension
        radius = min(3, shape[0]//2)
        coords = []
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                for k in range(-radius, radius+1):
                    xx = torch.clamp(px + i, 0, shape[0]-1)
                    yy = torch.clamp(py + j, 0, shape[1]-1)
                    zz = torch.clamp(pz + k, 0, shape[2]-1)
                    coords.append(universe_field[xx, yy, zz])
        coords = torch.stack(coords)
        # Then reduce it to dimension size by some means (like pooling)
        if coords.shape[0] > self.dimension:
            # pick random or top slice
            coords = coords[:self.dimension]
        elif coords.shape[0] < self.dimension:
            # pad
            pad_len = self.dimension - coords.shape[0]
            coords = torch.cat([coords, torch.zeros(pad_len, device=self.device)])
        return coords

########################################
# 3) Adaptive Field System
########################################
class AdaptiveFieldSystem:
    """
    Manages multiple AdaptiveTensorUnits, updating them each step.
    """
    def __init__(self, num_units=10, dimension=128, device='cpu'):
        self.units = [AdaptiveTensorUnit(dimension, device) for _ in range(num_units)]
        self.device = device

    def update(self, universe_field):
        stabilities = []
        for unit in self.units:
            stability = unit.interact_with_field(universe_field)
            stabilities.append(stability)
        return np.mean(stabilities)

########################################
# 4) PhysicalTensorSingularity
########################################
class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0, device='cpu'):
        self.dimension = dimension
        self.device = device
        if position is not None:
            self.position = position.clone().detach().float().to(device)
        else:
            self.position = torch.tensor(np.random.rand(3) * 50, dtype=torch.float32, device=device)
        self.velocity = torch.randn(3, device=device) * 0.1
        self.mass = mass
        self.core = torch.randn(dimension, device=device)
        self.field = self.generate_gravitational_field()

    def generate_gravitational_field(self):
        field = self.core.clone()
        r = torch.linspace(0, 2*np.pi, self.dimension, device=self.device)
        field *= torch.exp(-r / (self.mass+1e-6))
        return field

    def update_position(self, dt, force):
        accel = force / (self.mass+1e-6)
        self.velocity += accel * dt
        self.position += self.velocity * dt

########################################
# 5) PhysicalTensorUniverse
########################################
class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=10, dimension=128, device='cpu'):
        self.G = 6.67430e-11
        self.size = size
        self.dimension = dimension
        self.device = device
        self.space = torch.zeros((size, size, size), device=device)
        self.singularities = []
        self.initialize_singularities(num_singularities)

        # Attach an AdaptiveFieldSystem
        self.adaptive_system = AdaptiveFieldSystem(num_units=30, dimension=dimension, device=device)

    def initialize_singularities(self, num):
        self.singularities = []
        for _ in range(num):
            pos = torch.tensor(np.random.rand(3)*self.size, dtype=torch.float32, device=self.device)
            mass = np.random.exponential(1.0)
            self.singularities.append(
                PhysicalTensorSingularity(
                    dimension=self.dimension,
                    position=pos,
                    mass=mass,
                    device=self.device
                )
            )

    def update_tensor_interactions(self):
        # Grav. interactions
        positions = torch.stack([s.position for s in self.singularities])
        masses = torch.tensor([s.mass for s in self.singularities], device=self.device)
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = torch.norm(delta, dim=2) + 1e-10
        force_mag = self.G * masses.unsqueeze(1)*masses.unsqueeze(0)/(dist**2)
        force_dir = delta/dist.unsqueeze(2)
        net_force = torch.sum(force_mag.unsqueeze(2)*force_dir, dim=1)
        for i, sing in enumerate(self.singularities):
            sing.update_position(dt=0.1, force=net_force[i])

        # Then adaptive system
        stability = self.adaptive_system.update(self.space)
        return stability

    def update_space(self):
        self.space.fill_(0)
        x = torch.linspace(0, self.size, self.size, device=self.device)
        y = torch.linspace(0, self.size, self.size, device=self.device)
        z = torch.linspace(0, self.size, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        for s in self.singularities:
            R = torch.sqrt((X - s.position[0])**2 + (Y - s.position[1])**2 + (Z - s.position[2])**2)
            self.space += s.mass/(R+1) * torch.mean(s.field)

########################################
# 6) Visualization
########################################
def create_visualization(universe, colormap='gray'):
    slice_index = universe.size//2
    density_slice = universe.space[:, :, slice_index].cpu().numpy()
    dn = (density_slice - density_slice.min()) / (density_slice.max()-density_slice.min()+1e-8)
    # Convert to color
    cmap = plt.get_cmap(colormap)
    color_slice = cmap(dn)[:, :, :3]
    color_slice = (color_slice*255).astype(np.uint8)

    # Plot adaptive units in red
    # We'll only plot those whose z is near slice_index
    for unit in universe.adaptive_system.units:
        ux, uy, uz = unit.position.cpu().detach().numpy()
        if abs(uz - slice_index)<1.0:
            ix = int(np.clip(ux, 0, universe.size-1))
            iy = int(np.clip(uy, 0, universe.size-1))
            if 0<=ix<universe.size and 0<=iy<universe.size:
                color_slice[ix, iy] = [255,0,0]  # red

    pil_img = Image.fromarray(color_slice)
    pil_img = pil_img.resize((512,512))
    return pil_img

########################################
# 7) Pygame-based Menu and Renderer
########################################
class Button:
    def __init__(self, text, x, y, width, height, callback, font, bg_color=(70,70,70), text_color=(255,255,255)):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.callback = callback
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, (200,200,200), self.rect, 2)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Menu:
    def __init__(self, screen, renderer):
        self.screen = screen
        self.renderer = renderer
        self.width, self.height = screen.get_size()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        self.buttons = []
        self.current_menu = "main"

        self.color_schemes = list(self.renderer.color_schemes.keys())
        self.selected_color_scheme = self.renderer.selected_color_scheme
        self.resolutions = self.renderer.resolutions
        self.selected_resolution = self.renderer.selected_resolution

        self.setup_main_menu()

    def setup_main_menu(self):
        self.buttons = []
        w,h = 300,50
        spacing = 20
        start_x = (self.width - w)//2
        start_y = self.height//2 - (h+spacing)*2
        self.buttons.append(Button("Start Simulation", start_x, start_y, w, h, self.start_sim, self.font))
        self.buttons.append(Button("Restart Simulation", start_x, start_y+(h+spacing), w, h, self.restart_sim, self.font))
        self.buttons.append(Button("Settings", start_x, start_y+2*(h+spacing), w, h, self.open_settings, self.font))
        self.buttons.append(Button("Quit", start_x, start_y+3*(h+spacing), w, h, self.quit, self.font))

    def setup_settings_menu(self):
        self.buttons = []
        w,h = 220,40
        spacing=15
        sx = (self.width - w)//2
        sy = 100

        # Universe size
        self.buttons.append(Button(f"Increase Universe Size ({self.renderer.universe_size})",
                                   sx, sy, w, h, self.increase_universe_size, self.small_font))
        self.buttons.append(Button(f"Decrease Universe Size ({self.renderer.universe_size})",
                                   sx, sy+(h+spacing), w, h, self.decrease_universe_size, self.small_font))
        # singularities
        self.buttons.append(Button(f"Increase Singularities ({self.renderer.num_singularities})",
                                   sx, sy+2*(h+spacing), w, h, self.increase_singularities, self.small_font))
        self.buttons.append(Button(f"Decrease Singularities ({self.renderer.num_singularities})",
                                   sx, sy+3*(h+spacing), w, h, self.decrease_singularities, self.small_font))
        # color scheme
        self.buttons.append(Button(f"Color Scheme ({self.selected_color_scheme})",
                                   sx, sy+4*(h+spacing), w, h, self.change_color_scheme, self.small_font))
        # resolution
        res_str = f"{self.selected_resolution[0]}x{self.selected_resolution[1]}"
        self.buttons.append(Button(f"Resolution ({res_str})",
                                   sx, sy+5*(h+spacing), w, h, self.change_resolution, self.small_font))
        # back
        self.buttons.append(Button("Back", sx, sy+6*(h+spacing), w, h, self.back_main, self.small_font))

    def start_sim(self):
        self.current_menu = "simulation"
        logging.info("Simulation started.")
        if not self.renderer.sim_thread or not self.renderer.sim_thread.is_alive():
            self.renderer.start_simulation_thread()

    def restart_sim(self):
        if self.renderer.sim_thread and self.renderer.sim_thread.is_alive():
            self.renderer.stop_event.set()
            self.renderer.sim_thread.join()
        # reinit
        self.renderer.simulation.initialize_singularities(self.renderer.num_singularities)
        self.renderer.simulation.space.fill_(0)
        self.renderer.simulation.adaptive_system = AdaptiveFieldSystem(num_units=30, dimension=self.renderer.simulation.dimension, device=self.renderer.simulation.device)
        self.renderer.current_image = None
        self.renderer.stop_event.clear()
        self.renderer.start_simulation_thread()
        self.current_menu = "simulation"
        logging.info("Simulation restarted.")

    def open_settings(self):
        self.current_menu = "settings"
        self.setup_settings_menu()

    def quit(self):
        logging.info("Quit selected.")
        pygame.quit()
        sys.exit()

    def back_main(self):
        self.current_menu = "main"
        self.setup_main_menu()

    def increase_universe_size(self):
        self.renderer.universe_size += 10
        logging.info(f"Universe size increased to {self.renderer.universe_size}")
        self.setup_settings_menu()

    def decrease_universe_size(self):
        self.renderer.universe_size = max(10, self.renderer.universe_size-10)
        logging.info(f"Universe size decreased to {self.renderer.universe_size}")
        self.setup_settings_menu()

    def increase_singularities(self):
        self.renderer.num_singularities += 10
        logging.info(f"Num singularities: {self.renderer.num_singularities}")
        self.setup_settings_menu()

    def decrease_singularities(self):
        self.renderer.num_singularities = max(1, self.renderer.num_singularities-10)
        logging.info(f"Num singularities: {self.renderer.num_singularities}")
        self.setup_settings_menu()

    def change_color_scheme(self):
        idx = self.color_schemes.index(self.selected_color_scheme)
        idx = (idx+1)%len(self.color_schemes)
        self.selected_color_scheme = self.color_schemes[idx]
        self.renderer.selected_color_scheme = self.selected_color_scheme
        logging.info(f"Color scheme changed to {self.selected_color_scheme}")
        self.setup_settings_menu()

    def change_resolution(self):
        idx = self.resolutions.index(self.selected_resolution)
        idx = (idx+1)%len(self.resolutions)
        self.selected_resolution = self.resolutions[idx]
        self.renderer.selected_resolution = self.selected_resolution
        self.renderer.screen = pygame.display.set_mode(self.selected_resolution, pygame.RESIZABLE)
        logging.info(f"Resolution changed to {self.selected_resolution}")
        self.setup_settings_menu()

    def draw(self):
        self.screen.fill((0,0,0))
        for b in self.buttons:
            b.draw(self.screen)
        pygame.display.flip()

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            for b in self.buttons:
                if b.is_clicked(event.pos):
                    b.callback()

class SimulationRenderer:
    def __init__(self, universe_size=50, num_singularities=10, dimension=128, screen_size=(1024,768)):
        self.universe_size = universe_size
        self.num_singularities = num_singularities
        self.dimension = dimension

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.simulation = PhysicalTensorUniverse(size=universe_size, 
                                                 num_singularities=num_singularities,
                                                 dimension=dimension,
                                                 device=self.device)
        self.current_image = None
        self.lock = torch.multiprocessing.Lock()  # or threading.Lock()
        self.stop_event = torch.multiprocessing.Event()  # or threading.Event()
        self.sim_thread = None

        # Pygame
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
        pygame.display.set_caption("Adaptive Universe + Menu")
        self.clock = pygame.time.Clock()
        self.running = True

        # color schemes
        self.color_schemes = {
            "Gray": "gray",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Cividis": "cividis"
        }
        self.selected_color_scheme = "Gray"

        # resolutions
        self.resolutions = [(800,600),(1024,768),(1280,720),(1920,1080)]
        self.selected_resolution = (1024,768)

    def update_image(self, img):
        with self.lock:
            self.current_image = img

    def start_simulation_thread(self):
        """Start the simulation thread (fixed to use threading instead of multiprocessing)."""
        if self.sim_thread and self.sim_thread.is_alive():
            return  # Prevent multiple threads

        self.sim_thread = threading.Thread(
            target=simulation_thread_function,
            args=(self,),
            daemon=True
        )
        self.sim_thread.start()

    def draw(self):
        with self.lock:
            if self.current_image:
                mode = self.current_image.mode
                size = self.current_image.size
                data = self.current_image.tobytes()
                try:
                    pygame_image = pygame.image.fromstring(data, size, mode)
                except Exception as e:
                    logging.error(f"Image conversion error: {e}")
                    return
                # scale to screen
                pygame_image = pygame.transform.scale(pygame_image, self.screen.get_size())
                self.screen.blit(pygame_image, (0,0))

        pygame.display.flip()
        self.clock.tick(30)

########################################
# 8) Simulation Thread Function
########################################
def simulation_thread_function(renderer):
    logging.info("Simulation thread started.")
    print("Simulation thread started.")
    while not renderer.stop_event.is_set():
        stability = renderer.simulation.update_tensor_interactions()
        renderer.simulation.update_space()
        img = create_visualization(renderer.simulation, colormap=renderer.color_schemes[renderer.selected_color_scheme])
        renderer.update_image(img)
        logging.info(f"Simulation step done. Stability={stability:.4f}")
        time.sleep(0.1)
    logging.info("Simulation thread stopped.")
    print("Simulation thread stopped.")

########################################
# 9) Main Loop
########################################
def main():
    # Create renderer
    renderer = SimulationRenderer(universe_size=50, num_singularities=10, dimension=128, screen_size=(1024,768))
    menu = Menu(renderer.screen, renderer)

    # Main loop
    while renderer.running:
        for event in pygame.event.get():
            if event.type == QUIT:
                renderer.running = False
                if renderer.sim_thread and renderer.sim_thread.is_alive():
                    renderer.stop_event.set()
                    renderer.sim_thread.join()
            elif event.type == KEYDOWN:
                if menu.current_menu == "simulation":
                    # Any key -> back to main menu
                    menu.current_menu = "main"
                    menu.setup_main_menu()
                    if renderer.sim_thread and renderer.sim_thread.is_alive():
                        renderer.stop_event.set()
                        renderer.sim_thread.join()
                        renderer.stop_event.clear()
                else:
                    # Possibly ESC -> quit
                    if event.key == K_ESCAPE:
                        menu.quit()
                    elif event.key == K_f and menu.current_menu=="simulation":
                        # toggle fullscreen
                        pass
            elif event.type == MOUSEBUTTONDOWN:
                if menu.current_menu != "simulation":
                    menu.handle_event(event)
        # Draw
        if menu.current_menu == "simulation":
            renderer.draw()
        else:
            menu.draw()
    # Clean up
    if renderer.sim_thread and renderer.sim_thread.is_alive():
        renderer.stop_event.set()
        renderer.sim_thread.join()
    pygame.quit()
    sys.exit()

if __name__=="__main__":
    main()
