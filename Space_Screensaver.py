import numpy as np
import torch
import pygame
from pygame.locals import *
from scipy.ndimage import label
from PIL import Image
import threading
import time
import logging
import matplotlib.pyplot as plt  # For color maps
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, filename='simulation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0, device='cpu'):
        self.dimension = dimension
        self.device = device
        # Physical properties
        if position is not None:
            if isinstance(position, np.ndarray):
                self.position = torch.from_numpy(position).float().to(self.device)
            else:
                # If position is already a tensor, clone and detach it
                self.position = position.clone().detach().float().to(self.device)
        else:
            self.position = torch.tensor(np.random.rand(3), dtype=torch.float32, device=self.device)
        self.velocity = torch.randn(3, device=self.device) * 0.1
        self.mass = mass
        # Tensor properties
        self.core = torch.randn(dimension, device=self.device)
        self.field = self.generate_gravitational_field()

    def generate_gravitational_field(self):
        """Generate gravitational field based on mass"""
        field = self.core.clone()
        # Apply gravitational influence
        r = torch.linspace(0, 2 * np.pi, self.dimension, device=self.device)
        field *= torch.exp(-r / self.mass)  # Gravitational falloff
        return field

    def update_position(self, dt, force):
        """Update position using Newtonian physics"""
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt


class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=100, dimension=128, device='cpu'):
        self.G = 6.67430e-11  # Gravitational constant
        self.size = size
        self.dimension = dimension
        self.device = device
        self.space = torch.zeros((size, size, size), device=self.device)
        self.singularities = []
        self.initialize_singularities(num_singularities)

    def initialize_singularities(self, num):
        """Initialize singularities with random positions and masses"""
        self.singularities = []  # Reset list
        for _ in range(num):
            position = torch.tensor(np.random.rand(3) * self.size, dtype=torch.float32, device=self.device)
            mass = torch.distributions.Exponential(1.0).sample().item()  # Random masses
            self.singularities.append(
                PhysicalTensorSingularity(
                    dimension=self.dimension,
                    position=position,
                    mass=mass,
                    device=self.device
                )
            )

    def calculate_gravity(self, pos1, pos2, m1, m2):
        """Calculate gravitational force between two points"""
        r = pos2 - pos1
        distance = torch.norm(r) + 1e-10
        force_magnitude = self.G * m1 * m2 / (distance ** 2)
        return force_magnitude * r / distance

    def update_tensor_interactions(self):
        """Update tensor field interactions using vectorized operations"""
        positions = torch.stack([s.position for s in self.singularities])  # Shape: [N, 3]
        masses = torch.tensor([s.mass for s in self.singularities], device=self.device)  # Shape: [N]

        # Calculate pairwise distances and forces
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape: [N, N, 3]
        distance = torch.norm(delta, dim=2) + 1e-10  # Shape: [N, N]
        force_magnitude = self.G * masses.unsqueeze(1) * masses.unsqueeze(0) / (distance ** 2)  # Shape: [N, N]
        force_direction = delta / distance.unsqueeze(2)  # Shape: [N, N, 3]
        force = torch.sum(force_magnitude.unsqueeze(2) * force_direction, dim=1)  # Shape: [N, 3]

        # Apply tensor field interactions
        fields = torch.stack([s.field for s in self.singularities])  # Shape: [N, D]
        field_interaction = torch.tanh(torch.matmul(fields, fields.T))  # Shape: [N, N]
        force *= (1 + torch.mean(field_interaction, dim=1)).unsqueeze(1)

        # Update positions
        for i, singularity in enumerate(self.singularities):
            singularity.update_position(dt=0.1, force=force[i])

    def update_space(self):
        """Update 3D space based on singularity positions and fields"""
        self.space.fill_(0)
        x = torch.linspace(0, self.size, self.size, device=self.device)
        y = torch.linspace(0, self.size, self.size, device=self.device)
        z = torch.linspace(0, self.size, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        for s in self.singularities:
            # Calculate distance from singularity to each point
            R = torch.sqrt((X - s.position[0]) ** 2 +
                          (Y - s.position[1]) ** 2 +
                          (Z - s.position[2]) ** 2)
            # Add field influence
            self.space += s.mass / (R + 1) * torch.mean(s.field)

    def detect_structures(self):
        """Detect galaxy-like structures using clustering"""
        structures = []
        density_threshold = torch.mean(self.space) + torch.std(self.space)

        # Find high density regions
        dense_regions = self.space > density_threshold

        # Convert to NumPy for clustering
        dense_indices = torch.nonzero(dense_regions, as_tuple=False).cpu().numpy()

        if dense_indices.size == 0:
            return structures

        # Basic clustering using scipy
        labeled_array, num_features = label(dense_regions.cpu().numpy())

        for i in range(1, num_features + 1):
            region = torch.nonzero(labeled_array == i, as_tuple=False).numpy()
            if region.size == 0:
                continue
            center = np.mean(region, axis=0)
            mass = torch.sum(self.space[labeled_array == i]).item()
            size = region.shape[0]
            structures.append({
                'center': center,
                'mass': mass,
                'size': size
            })

        return structures


def create_density_slice_image(universe, slice_axis='z'):
    """
    Create a 2D density slice image from the 3D space.

    Args:
        universe (PhysicalTensorUniverse): The simulation universe.
        slice_axis (str): The axis to slice ('x', 'y', or 'z').

    Returns:
        PIL.Image: The rendered density slice image.
    """
    # Select the slice axis and compute the middle slice
    size = universe.size
    if slice_axis == 'x':
        slice_index = size // 2
        density_slice = universe.space[slice_index, :, :].cpu().numpy()
    elif slice_axis == 'y':
        slice_index = size // 2
        density_slice = universe.space[:, slice_index, :].cpu().numpy()
    else:  # 'z'
        slice_index = size // 2
        density_slice = universe.space[:, :, slice_index].cpu().numpy()

    # Normalize the density slice for visualization
    density_normalized = (density_slice - density_slice.min()) / (density_slice.max() - density_slice.min())
    density_normalized = np.uint8(255 * density_normalized)

    # Apply a color map for better visualization
    density_colored = Image.fromarray(density_normalized).convert("L")
    density_colored = density_colored.convert("RGB")
    density_colored = density_colored.resize((1024, 1024))  # Stretch to 1024x1024

    return density_colored


class Button:
    def __init__(self, text, x, y, width, height, callback, font, bg_color=(70, 70, 70), text_color=(255, 255, 255)):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.callback = callback
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)  # Border
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class Menu:
    def __init__(self, screen, renderer):
        self.screen = screen
        self.renderer = renderer
        self.width, self.height = self.screen.get_size()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        self.buttons = []
        self.current_menu = "main"  # Options: main, settings, help, credits

        self.color_schemes = list(renderer.color_schemes.keys())
        self.selected_color_scheme = renderer.selected_color_scheme

        self.resolutions = renderer.resolutions
        self.selected_resolution = renderer.selected_resolution

        self.setup_main_menu()

    def setup_main_menu(self):
        self.buttons = []
        button_width = 300
        button_height = 50
        spacing = 20
        start_x = (self.width - button_width) // 2
        start_y = self.height // 2 - (button_height + spacing) * 2

        self.buttons.append(Button("Start Simulation", start_x, start_y, button_width, button_height, self.start_simulation, self.font))
        self.buttons.append(Button("Restart Simulation", start_x, start_y + (button_height + spacing), button_width, button_height, self.restart_simulation, self.font))
        self.buttons.append(Button("Settings", start_x, start_y + 2*(button_height + spacing), button_width, button_height, self.open_settings, self.font))
        self.buttons.append(Button("Help", start_x, start_y + 3*(button_height + spacing), button_width, button_height, self.open_help, self.font))
        self.buttons.append(Button("Credits", start_x, start_y + 4*(button_height + spacing), button_width, button_height, self.open_credits, self.font))
        self.buttons.append(Button("Quit", start_x, start_y + 5*(button_height + spacing), button_width, button_height, self.quit, self.font))

    def setup_settings_menu(self):
        self.buttons = []
        button_width = 200
        button_height = 40
        spacing = 15
        start_x = self.width // 4
        start_y = 100

        self.buttons.append(Button("Increase Universe Size", start_x, start_y, button_width, button_height, self.increase_universe_size, self.small_font))
        self.buttons.append(Button("Decrease Universe Size", start_x + 220, start_y, button_width, button_height, self.decrease_universe_size, self.small_font))

        self.buttons.append(Button("Increase Singularities", start_x, start_y + 60, button_width, button_height, self.increase_singularities, self.small_font))
        self.buttons.append(Button("Decrease Singularities", start_x + 220, start_y + 60, button_width, button_height, self.decrease_singularities, self.small_font))

        self.buttons.append(Button(f"Change Color Scheme ({self.selected_color_scheme})", start_x, start_y + 120, button_width + 220, button_height, self.change_color_scheme, self.small_font))
        self.buttons.append(Button(f"Change Resolution ({self.selected_resolution[0]}x{self.selected_resolution[1]})", start_x, start_y + 180, button_width + 220, button_height, self.change_resolution, self.small_font))

        self.buttons.append(Button("Back to Main Menu", start_x + 100, start_y + 240, 200, button_height, self.back_to_main, self.small_font))

    def setup_help_menu(self):
        self.buttons = []
        button_width = 200
        button_height = 40
        spacing = 20
        start_x = (self.width - button_width) // 2
        start_y = self.height - 100

        self.buttons.append(Button("Back to Main Menu", start_x, start_y, button_width, button_height, self.back_to_main, self.small_font))

    def setup_credits_menu(self):
        self.buttons = []
        button_width = 200
        button_height = 40
        spacing = 20
        start_x = (self.width - button_width) // 2
        start_y = self.height - 100

        self.buttons.append(Button("Back to Main Menu", start_x, start_y, button_width, button_height, self.back_to_main, self.small_font))

    def start_simulation(self):
        self.current_menu = "simulation"
        logging.info("Simulation started.")
        print("Simulation started.")

    def restart_simulation(self):
        if self.renderer.simulation_thread and self.renderer.simulation_thread.is_alive():
            self.renderer.stop_event.set()
            self.renderer.simulation_thread.join()
            logging.info("Simulation stopped for restart.")
            print("Simulation stopped for restart.")
        # Reinitialize singularities with current settings
        self.renderer.simulation.initialize_singularities(self.renderer.num_singularities)
        # Clear current image
        self.renderer.update_image(None)
        # Restart simulation thread
        self.renderer.stop_event.clear()
        self.renderer.simulation_thread = threading.Thread(target=simulation_thread_function, args=(self.renderer,), daemon=True)
        self.renderer.simulation_thread.start()
        self.current_menu = "simulation"
        logging.info("Simulation restarted.")
        print("Simulation restarted.")

    def open_settings(self):
        self.current_menu = "settings"
        self.setup_settings_menu()

    def open_help(self):
        self.current_menu = "help"
        self.setup_help_menu()

    def open_credits(self):
        self.current_menu = "credits"
        self.setup_credits_menu()

    def quit(self):
        logging.info("Application quit by user.")
        pygame.quit()
        sys.exit()

    def back_to_main(self):
        self.current_menu = "main"
        self.setup_main_menu()

    def increase_universe_size(self):
        self.renderer.universe_size += 10
        logging.info(f"Universe Size increased to {self.renderer.universe_size}")
        print(f"Universe Size increased to {self.renderer.universe_size}")

    def decrease_universe_size(self):
        self.renderer.universe_size = max(10, self.renderer.universe_size - 10)
        logging.info(f"Universe Size decreased to {self.renderer.universe_size}")
        print(f"Universe Size decreased to {self.renderer.universe_size}")

    def increase_singularities(self):
        self.renderer.num_singularities += 10
        logging.info(f"Number of Singularities increased to {self.renderer.num_singularities}")
        print(f"Number of Singularities increased to {self.renderer.num_singularities}")

    def decrease_singularities(self):
        self.renderer.num_singularities = max(10, self.renderer.num_singularities - 10)
        logging.info(f"Number of Singularities decreased to {self.renderer.num_singularities}")
        print(f"Number of Singularities decreased to {self.renderer.num_singularities}")

    def change_color_scheme(self):
        current_index = self.color_schemes.index(self.selected_color_scheme)
        next_index = (current_index + 1) % len(self.color_schemes)
        self.selected_color_scheme = self.color_schemes[next_index]
        self.renderer.selected_color_scheme = self.selected_color_scheme
        logging.info(f"Color Scheme changed to {self.selected_color_scheme}")
        print(f"Color Scheme changed to {self.selected_color_scheme}")
        self.setup_settings_menu()  # Update button text

    def change_resolution(self):
        current_index = self.resolutions.index(self.selected_resolution)
        next_index = (current_index + 1) % len(self.resolutions)
        self.selected_resolution = self.resolutions[next_index]
        self.renderer.selected_resolution = self.selected_resolution
        self.screen = pygame.display.set_mode(self.selected_resolution, pygame.RESIZABLE)
        logging.info(f"Resolution changed to {self.selected_resolution}")
        print(f"Resolution changed to {self.selected_resolution}")
        self.setup_settings_menu()  # Update button text

    def draw(self):
        self.screen.fill((0, 0, 0))  # Black background
        for button in self.buttons:
            button.draw(self.screen)

        # Draw help text if in help menu
        if self.current_menu == "help":
            help_text = [
                "Space Screensaver Help",
                "",
                "Controls:",
                "- Press any key during the simulation to open the menu and pause.",
                "- In the menu:",
                "  - Start Simulation: Run the simulation in live mode.",
                "  - Restart Simulation: Restart the simulation with current settings.",
                "  - Adjust Universe Size: Determines spatial dimensions.",
                "  - Adjust Singularities: Number of entities affecting the density field.",
                "  - Change Color Scheme: Switch between predefined color schemes.",
                "  - Change Resolution: Select window size.",
                "  - Credits: View application credits.",
                "  - Quit: Exit the application.",
                "",
                "Color Schemes:",
                "- Black & White: Classic monochrome visualization.",
                "- Viridis, Plasma, Inferno, Magma, Cividis: Color maps for enhanced visualization.",
                "",
                "Press 'F' to toggle full-screen mode during simulation."
            ]

            for i, line in enumerate(help_text):
                text_surf = self.small_font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surf, (50, 50 + i * 25))

        # Draw credits if in credits menu
        if self.current_menu == "credits":
            credits_text = [
                "Space Screensaver Credits",
                "",
                "Developed by:",
                "- Antti Luode",
                "- ChatGPT",
                "- ClaudeAI",
                "",
                "Thank you for using the Space Screensaver!"
            ]

            for i, line in enumerate(credits_text):
                text_surf = self.small_font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surf, (50, 50 + i * 25))

        pygame.display.flip()


class SimulationRenderer:
    def __init__(self, universe_size, num_singularities, screen_size=(1024, 768)):
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        pygame.display.set_caption("Space Screensaver - Fast Tensor Universe Simulation")
        self.clock = pygame.time.Clock()
        self.running = True

        # Full-screen flag
        self.fullscreen = False

        # Simulation parameters
        self.universe_size = universe_size
        self.num_singularities = num_singularities

        # Initialize simulation
        self.simulation = PhysicalTensorUniverse(
            size=self.universe_size,
            num_singularities=self.num_singularities,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Shared data
        self.current_image = None
        self.lock = threading.Lock()

        # Color scheme
        self.color_schemes = {
            "Black & White": "gray",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Cividis": "cividis"
        }
        self.selected_color_scheme = "Black & White"

        # Resolution
        self.resolutions = [(800, 600), (1024, 768), (1280, 720), (1920, 1080)]
        self.selected_resolution = (1024, 768)

        # Simulation thread and control
        self.simulation_thread = None
        self.stop_event = threading.Event()

    def initialize_simulation_thread(self):
        """Initialize and start the simulation thread."""
        self.simulation_thread = threading.Thread(target=simulation_thread_function, args=(self,), daemon=True)
        self.simulation_thread.start()

    def update_image(self, image):
        with self.lock:
            self.current_image = image

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.selected_resolution, pygame.RESIZABLE)

    def draw_parameters(self):
        font = pygame.font.SysFont(None, 24)
        text_color = (255, 255, 255)  # White

        params_text = f"Universe Size: {self.universe_size} | Singularities: {self.num_singularities} | Color Scheme: {self.selected_color_scheme}"
        text_surface = font.render(params_text, True, text_color)
        self.screen.blit(text_surface, (10, 10))

    def draw(self):
        with self.lock:
            if self.current_image:
                # Apply selected color scheme using matplotlib
                density_array = np.array(self.current_image.convert("L"))
                cmap = plt.get_cmap(self.color_schemes[self.selected_color_scheme])
                colored_density = cmap(density_array / 255.0)[:, :, :3]  # Ignore alpha
                colored_density = (colored_density * 255).astype(np.uint8)
                density_colored_image = Image.fromarray(colored_density)

                # Convert PIL Image to Pygame Surface
                mode = density_colored_image.mode
                size = density_colored_image.size
                data = density_colored_image.tobytes()
                try:
                    pygame_image = pygame.image.fromstring(data, size, mode)
                except Exception as e:
                    logging.error(f"Error converting image to Pygame surface: {e}")
                    return

                # Scale image to fit the screen
                pygame_image = pygame.transform.scale(pygame_image, self.screen.get_size())
                self.screen.blit(pygame_image, (0, 0))

                # Draw parameters
                self.draw_parameters()

        pygame.display.flip()


def simulation_thread_function(renderer):
    """
    Function to run the simulation in a separate thread.
    Supports live mode by running indefinitely until stop_event is set.
    """
    logging.info("Simulation thread started.")
    print("Simulation thread started.")

    while not renderer.stop_event.is_set():
        renderer.simulation.update_tensor_interactions()
        renderer.simulation.update_space()

        # Capture density slice image
        img = create_density_slice_image(renderer.simulation, slice_axis='z')
        renderer.update_image(img)

        # Log progress periodically
        logging.info("Simulation step completed.")
        print("Simulation step completed.")

        # Optional: Sleep to control simulation speed
        time.sleep(0.1)  # Adjust as needed

    logging.info("Simulation thread stopped.")
    print("Simulation thread stopped.")


def main():
    # Simulation parameters
    universe_size = 100  # Default value
    num_singularities = 200  # Default value

    # Initialize renderer
    renderer = SimulationRenderer(
        universe_size=universe_size,
        num_singularities=num_singularities,
        screen_size=(1024, 768)
    )

    # Initialize menu
    menu = Menu(renderer.screen, renderer)

    # Run the main loop
    while renderer.running:
        for event in pygame.event.get():
            if event.type == QUIT:
                renderer.running = False
                if renderer.simulation_thread and renderer.simulation_thread.is_alive():
                    renderer.stop_event.set()
                    renderer.simulation_thread.join()
            elif event.type == KEYDOWN:
                if menu.current_menu == "simulation":
                    # Any key press during simulation brings up the menu and pauses
                    menu.current_menu = "main"
                    menu.setup_main_menu()
                    if renderer.simulation_thread and renderer.simulation_thread.is_alive():
                        renderer.stop_event.set()
                        renderer.simulation_thread.join()
                        renderer.stop_event.clear()
                else:
                    # Pass event to menu
                    if event.key == K_ESCAPE:
                        menu.quit()
                    elif event.key == K_f:
                        renderer.toggle_fullscreen()
            elif event.type == MOUSEBUTTONDOWN:
                if menu.current_menu != "simulation":
                    for button in menu.buttons:
                        if button.is_clicked(event.pos):
                            button.callback()
                            # If "Start Simulation" was clicked, start the simulation thread
                            if button.text == "Start Simulation":
                                if not renderer.simulation_thread or not renderer.simulation_thread.is_alive():
                                    renderer.initialize_simulation_thread()
                                    menu.current_menu = "simulation"
                            # If "Restart Simulation" was clicked, restart the simulation
                            if button.text == "Restart Simulation":
                                if renderer.simulation_thread and renderer.simulation_thread.is_alive():
                                    renderer.stop_event.set()
                                    renderer.simulation_thread.join()
                                # Reinitialize singularities with current settings
                                renderer.simulation.initialize_singularities(renderer.num_singularities)
                                # Clear current image
                                renderer.update_image(None)
                                # Restart simulation thread
                                renderer.stop_event.clear()
                                renderer.initialize_simulation_thread()
                                menu.current_menu = "simulation"

        if menu.current_menu != "simulation":
            menu.draw()
        else:
            renderer.draw()

        renderer.clock.tick(60)  # Limit to 60 FPS

    # Wait for simulation thread to finish before exiting
    if renderer.simulation_thread and renderer.simulation_thread.is_alive():
        renderer.stop_event.set()
        renderer.simulation_thread.join()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
