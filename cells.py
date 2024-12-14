import numpy as np
import torch
import pygame
from pygame.locals import *
from scipy.ndimage import label
from PIL import Image
import io
import tempfile
import threading
import time
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime

sim_thread = None  # Global variable for simulation thread

# Configure logging
logging.basicConfig(level=logging.INFO, filename='simulation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class LatentSpaceProjector:
    """Handles projections between latent space and manifested reality"""
    def __init__(self, dimension=128, layers=3, device='cpu'):
        self.dimension = dimension
        self.device = device
        self.layers = layers
        # Initialize projection tensors
        self.reality_tensors = [
            torch.randn(dimension, device=device) for _ in range(layers)
        ]
        self.quantum_noise = torch.randn(dimension, device=device) * 0.01
        
    def project_to_reality(self, latent_tensor):
        """Project from latent space to reality through tensor layers"""
        reality = latent_tensor
        for layer in self.reality_tensors:
            reality = torch.tanh(reality * layer + self.quantum_noise)
        return reality
    
    def inject_quantum_fluctuations(self):
        """Simulate quantum fluctuations in latent space"""
        self.quantum_noise = torch.randn_like(self.quantum_noise) * 0.01

class TimeProjector:
    """Handles the projection of time from tensor interactions"""
    def __init__(self, dimension=128, device='cpu'):
        self.dimension = dimension
        self.device = device
        self.time_tensor = torch.randn(dimension, device=device)
        self.flow_rate = torch.ones(1, device=device)
        
    def calculate_time_dilation(self, gravity_strength):
        """Calculate time dilation based on gravitational strength"""
        if isinstance(gravity_strength, (int, float)):
            gravity_strength = torch.tensor(gravity_strength, device=self.device)
        gravity_strength = torch.as_tensor(gravity_strength, device=self.device)
        return 1.0 / torch.sqrt(1.0 + 2.0 * gravity_strength)
    
    def project_time_field(self, space_tensor):
        """Project time field based on space tensor"""
        return torch.tanh(space_tensor * self.time_tensor) * self.flow_rate

class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0, device='cpu'):
        self.dimension = dimension
        self.device = device
        
        # Physical properties
        if position is not None:
            if isinstance(position, np.ndarray):
                self.position = torch.from_numpy(position).float().to(self.device)
            else:
                self.position = position.clone().detach().float().to(self.device)
        else:
            self.position = torch.tensor(np.random.rand(3), dtype=torch.float32, device=self.device)
        
        # Enhanced core tensor properties - these are the reality projectors
        self.mass = mass * 5.0  # Increased mass influence
        self.core_tensor = torch.randn(dimension, device=self.device) * 4.0  # Stronger core field
        self.quantum_state = torch.randn(dimension, device=self.device) * 2.0
        
        # Movement properties
        self.velocity = torch.randn(3, device=self.device) * 0.8
        self.angular_velocity = torch.randn(1, device=self.device) * 0.3
        self.angle = torch.zeros(1, device=self.device)
        self.orbit_center = self.position.clone()
        self.orbit_radius = torch.rand(1, device=self.device) * 12.0
        
        # Reality projection fields
        self.field = self.generate_reality_field()
        self.latent_projector = LatentSpaceProjector(dimension, layers=5, device=device)  # More projection layers
        
    def generate_reality_field(self):
        field = self.core_tensor.clone()
        r = torch.linspace(0, 2 * np.pi, self.dimension, device=self.device)
        projection = torch.exp(-r / (self.mass * 3.0))
        field_strength = 1.0 / (1.0 + r * 0.1)
        field = field * projection * field_strength * 5.0
        return (field - field.min()) / (field.max() - field.min() + 1e-8)  # Normalize
        
   
    def update_position(self, dt, force):
        # Project quantum state into reality
        projected_state = self.latent_projector.project_to_reality(self.quantum_state) * 3.0
        self.quantum_state = projected_state
        
        # Enhanced orbital motion
        self.angle += self.angular_velocity * dt * 1.5
        
        # Calculate position with stronger dynamics
        orbital_position = self.orbit_center + torch.tensor([
            torch.cos(self.angle) * self.orbit_radius * 1.5,
            torch.sin(self.angle) * self.orbit_radius * 1.5,
            torch.sin(self.angle * 0.7) * self.orbit_radius * 0.6
        ], device=self.device)
        
        # Update movement
        acceleration = force / self.mass
        self.velocity += acceleration * dt * 1.5
        self.position = orbital_position + self.velocity * dt
        self.orbit_center += self.velocity * dt * 0.5
        
        # Update reality projection
        tensor_manifestation = torch.sin(self.angle) * 0.5 + 1.0
        self.field = self.generate_reality_field() * tensor_manifestation * 2.0


class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=100, dimension=128, device='cpu'):
        self.G = 6.67430e-11 * 1e13  # Increased interaction strength
        self.size = size
        self.dimension = dimension
        self.device = device
        
        # Initialize space tensors
        self.space = torch.zeros((size, size, size), device=self.device)
        self.reality_tensor = torch.randn((size, size, size), device=device) * 2.0
        self.quantum_field = torch.randn((size, size, size), device=device) * 1.0
        
        # Reality projection components
        self.time_projector = TimeProjector(dimension, device=device)
        self.latent_space = torch.randn((size, size, size), device=device) * 2.0
        self.reality_field = torch.randn((size, size, size), device=device) * 1.5
        
        # Initialize tensor singularities
        self.singularities = []
        self.initialize_singularities(num_singularities)

    def update_tensor_interactions(self):
        try:
            # Enhanced quantum fluctuations
            self.quantum_field = torch.randn_like(self.quantum_field) * 0.1  # Reduced fluctuation
            
            # Gather tensor states
            positions = torch.stack([s.position for s in self.singularities])
            masses = torch.tensor([s.mass for s in self.singularities], device=self.device)
            
            # Calculate tensor field interactions
            delta = positions.unsqueeze(1) - positions.unsqueeze(0)
            distance = torch.norm(delta, dim=2) + 1e-10
            
            # Reduced interaction strength
            force_magnitude = self.G * masses.unsqueeze(1) * masses.unsqueeze(0) / (distance ** 1.8)
            force_direction = delta / distance.unsqueeze(2)
            force = torch.sum(force_magnitude.unsqueeze(2) * force_direction, dim=1)
            
            # Reduce field interaction strength
            fields = torch.stack([s.field for s in self.singularities])
            field_interaction = torch.tanh(torch.matmul(fields, fields.T))
            quantum_influence = torch.mean(self.quantum_field)
            
            force *= (1 + torch.mean(field_interaction, dim=1) + quantum_influence).unsqueeze(1)
            
            dt = 0.05  # Reduced time step
            
            for i, singularity in enumerate(self.singularities):
                singularity.update_position(dt, force[i])
                
        except Exception as e:
            print(f"Error in tensor interactions: {e}")
            time.sleep(0.1)

    def initialize_singularities(self, num):
            """Initialize tensor singularities with random positions and masses"""
            for _ in range(num):
                # Create random position within universe bounds
                position = torch.tensor(np.random.rand(3) * self.size, dtype=torch.float32, device=self.device)
                
                # Create random mass with exponential distribution for variety
                mass = torch.distributions.Exponential(2.0).sample().item()
                
                # Create new singularity
                self.singularities.append(
                    PhysicalTensorSingularity(
                        dimension=self.dimension,
                        position=position,
                        mass=mass,
                        device=self.device
                    )
                )

    def update_space(self):
        self.space.fill_(0)
        x = torch.linspace(0, self.size, self.size, device=self.device)
        y = torch.linspace(0, self.size, self.size, device=self.device)
        z = torch.linspace(0, self.size, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        # Project reality from latent space
        self.reality_field = torch.tanh(self.latent_space + self.quantum_field * 4.0)

        # Accumulate tensor field projections
        for s in self.singularities:
            R = torch.sqrt((X - s.position[0]) ** 2 +
                        (Y - s.position[1]) ** 2 +
                        (Z - s.position[2]) ** 2) + 0.1

            # Enhanced reality projection
            quantum_factor = 1.5 + torch.sin(self.time_projector.time_tensor[0])
            field_strength = s.mass / (R * 0.5) * torch.mean(s.field) * quantum_factor

            # Apply time dilation to field
            time_dilation = self.time_projector.calculate_time_dilation(field_strength)

            # Project combined fields into space
            self.space += (field_strength * time_dilation + self.reality_field) * 1.5

        # Final reality normalization with enhanced contrast
        self.space = torch.tanh(self.space * 3.0)


    def detect_structures(self):
        """Detect emergent structures with enhanced sensitivity"""
        structures = []
        density_threshold = torch.mean(self.space) + 0.8 * torch.std(self.space)
        
        # Convert to tensor first, then back to numpy for labeling
        dense_regions = (self.space > density_threshold).cpu().numpy()
        labeled_array, num_features = label(dense_regions)
        
        for i in range(1, num_features + 1):
            # Convert labeled_array back to tensor for comparison
            region_mask = torch.from_numpy(labeled_array == i).to(self.device)
            region_indices = torch.nonzero(region_mask)
            
            if region_indices.size(0) == 0:
                continue
            
            center = region_indices.float().mean(dim=0).cpu().numpy()
            mass = torch.sum(self.space[region_mask]).item()
            size = region_indices.size(0)
            
            # Get quantum coherence for the region
            quantum_coherence = torch.mean(self.quantum_field[region_mask]).item() * 2.0
            
            structures.append({
                'center': center,
                'mass': mass,
                'size': size,
                'quantum_coherence': quantum_coherence
            })
        
        return structures


class SimulationRenderer:
    def __init__(self, universe_size, num_singularities, simulation_steps, screen_size=(1024, 768)):
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        pygame.display.set_caption("Space Screensaver - Tensor Universe Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        self.fullscreen = False
        
        # Screenshot directory
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

        # Simulation parameters
        self.universe_size = universe_size
        self.num_singularities = num_singularities
        self.simulation_steps = simulation_steps
        
        # Display parameters
        self.font = pygame.font.SysFont(None, 24)
        
        # Color schemes dictionary
        self.color_schemes = {
            "Black & White": "gray",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Cividis": "cividis"
        }
        self.selected_color_scheme = "Plasma"
        
        # Resolution options
        self.resolutions = [(800, 600), (1024, 768), (1280, 720), (1920, 1080)]
        self.selected_resolution = screen_size
        
        # Initialize simulation
        self.simulation = PhysicalTensorUniverse(
            size=self.universe_size,
            num_singularities=self.num_singularities,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Shared data
        self.current_image = None
        self.lock = threading.Lock()
        
        # Physics display data
        self.avg_time_dilation = 1.0
        self.avg_quantum_coherence = 0.0
        self.num_structures = 0
        self.reality_density = 0.0

    def update_image(self, image):
        """Update the current display image thread-safely"""
        with self.lock:
            self.current_image = image

    def update_physics_data(self):
        """Update physics-related display data"""
        structures = self.simulation.detect_structures()
        self.num_structures = len(structures)
        
        if structures:
            self.avg_quantum_coherence = np.mean([s['quantum_coherence'] for s in structures])
        else:
            self.avg_quantum_coherence = 0.0
            
        # Calculate average time dilation from gravitational fields
        space_mean = torch.mean(self.simulation.space).item()
        self.avg_time_dilation = self.simulation.time_projector.calculate_time_dilation(space_mean).item()
        
        # Calculate reality density
        self.reality_density = torch.mean(self.simulation.reality_field).item()

    def draw_parameters(self):
        """Draw simulation parameters and physics information"""
        params = [
            f"Universe Size: {self.universe_size}",
            f"Singularities: {self.num_singularities}",
            f"Color Scheme: {self.selected_color_scheme}",
            f"Time Dilation: {self.avg_time_dilation:.3f}",
            f"Quantum Coherence: {self.avg_quantum_coherence:.3f}",
            f"Structures: {self.num_structures}",
            f"Reality Density: {self.reality_density:.3f}"
        ]
        
        y_offset = 10
        for param in params:
            text_surface = self.font.render(param, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

    def capture_screenshot(self):
        """Save current screen as an image file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"universe_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        pygame.image.save(self.screen, filepath)
        logging.info(f"Screenshot saved: {filepath}")
        print(f"Screenshot saved: {filepath}")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.selected_resolution, pygame.RESIZABLE)

    def draw(self):
        """Main rendering function"""
        with self.lock:
            if self.current_image:
                # Apply selected color scheme
                density_array = np.array(self.current_image.convert("L"))
                cmap = plt.get_cmap(self.color_schemes[self.selected_color_scheme])
                colored_density = cmap(density_array / 255.0)[:, :, :3]
                colored_density = (colored_density * 255).astype(np.uint8)
                density_colored_image = Image.fromarray(colored_density)

                # Convert to Pygame surface
                mode = density_colored_image.mode
                size = density_colored_image.size
                data = density_colored_image.tobytes()
                pygame_image = pygame.image.fromstring(data, size, mode)

                # Scale to screen
                pygame_image = pygame.transform.scale(pygame_image, self.screen.get_size())
                self.screen.blit(pygame_image, (0, 0))

                # Update and draw parameters
                self.update_physics_data()
                self.draw_parameters()

        pygame.display.flip()

    def handle_keypress(self, event):
        """Handle keyboard input"""
        if event.key == K_SPACE:
            return "menu"
        elif event.key == K_f:
            self.toggle_fullscreen()
            return "continue"
        elif event.key == K_c:
            self.capture_screenshot()
            return "continue"
        return "continue"

    def reinitialize_simulation(self):
        """Reinitialize the simulation with current parameters"""
        self.simulation = PhysicalTensorUniverse(
            size=self.universe_size,
            num_singularities=self.num_singularities,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def cleanup(self):
        """Cleanup resources"""
        pygame.quit()

from PIL import Image
import numpy as np


def create_density_slice_image(universe, slice_axis='z'):
    """
    Create a 2D density slice image with enhanced contrast and visual effects,
    with proper handling of edge cases.
    """
    try:
        # Get the universe size
        size = universe.size

        # Select the slice based on the chosen axis
        if slice_axis == 'x':
            slice_index = size // 2
            density_slice = universe.space[slice_index, :, :].cpu().numpy()
        elif slice_axis == 'y':
            slice_index = size // 2
            density_slice = universe.space[:, slice_index, :].cpu().numpy()
        else:  # 'z'
            slice_index = size // 2
            density_slice = universe.space[:, :, slice_index].cpu().numpy()

        # Handle NaN and infinite values first
        density_slice = np.nan_to_num(density_slice, nan=0.0, posinf=1.0, neginf=0.0)

        # Enhance contrast by scaling the values
        density_slice *= 3.0

        # Apply gamma correction for better visibility
        density_slice = np.power(np.abs(density_slice), 0.5) * np.sign(density_slice)

        # Add edge enhancement
        density_slice = np.gradient(density_slice)[0] + density_slice

        # Initial normalization
        density_min = np.min(density_slice)
        density_max = np.max(density_slice)

        # Check if the slice has any variation
        if np.isclose(density_max, density_min, rtol=1e-10, atol=1e-10):
            # If no variation, return a blank image
            density_normalized = np.zeros_like(density_slice)
        else:
            # Normalize to 0-1 range
            density_normalized = (density_slice - density_min) / (density_max - density_min)

            # Calculate percentiles for contrast stretching
            try:
                p2, p98 = np.percentile(density_normalized[~np.isnan(density_normalized)], (2, 98))
                if not np.isclose(p98, p2, rtol=1e-10, atol=1e-10):
                    density_normalized = (density_normalized - p2) / (p98 - p2)
            except (ValueError, IndexError):
                # If percentile calculation fails, use full range
                pass

        # Ensure values are in 0-1 range
        density_normalized = np.clip(density_normalized, 0, 1)

        # Convert to uint8 (0-255 range)
        density_normalized = (density_normalized * 255).astype(np.uint8)

        # Create PIL image
        density_colored = Image.fromarray(density_normalized).convert("L")
        density_colored = density_colored.convert("RGB")

        # Resize to standard dimensions for display
        density_colored = density_colored.resize((1024, 1024), Image.Resampling.LANCZOS)

        return density_colored

    except Exception as e:
        # If any step fails, return a blank image and log the error
        print(f"Error in create_density_slice_image: {e}")
        blank_image = Image.new('RGB', (1024, 1024), color='black')
        return blank_image

def simulation_thread_function(renderer):
    """
    Function to run the simulation continuously in a separate thread.
    Updates the simulation state and creates visualization frames.
    """
    frame_interval = 0.1  # Reduce to 10 FPS for better performance
    last_update = time.time()
    print("Simulation thread started")  # Debug output

    while renderer.running:
        try:
            # Update simulation physics
            renderer.simulation.update_tensor_interactions()
            renderer.simulation.update_space()

            # Create visualization at regular intervals
            current_time = time.time()
            if current_time - last_update >= frame_interval:
                # Generate density slice image
                img = create_density_slice_image(renderer.simulation, slice_axis='z')
                renderer.update_image(img)
                last_update = current_time
                print("Frame updated")  # Debug output

            # Longer sleep to reduce CPU usage
            time.sleep(0.05)

        except Exception as e:
            print(f"Error in simulation thread: {e}")
            time.sleep(1)  # Sleep longer on error

    print("Simulation thread stopped")

class Button:
    def __init__(self, text, x, y, width, height, callback, font, bg_color=(70, 70, 70), text_color=(255, 255, 255)):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.callback = callback
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, surface):
        # Draw button background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        # Draw button border
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        # Render text
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
        self.current_menu = "main"

        self.color_schemes = list(renderer.color_schemes.keys())
        self.selected_color_scheme = renderer.selected_color_scheme
        self.resolutions = renderer.resolutions
        self.selected_resolution = renderer.selected_resolution

        self.setup_main_menu()

    def setup_help_menu(self):
        self.buttons = []
        button_width = 200
        button_height = 40
        spacing = 20
        start_x = (self.width - button_width) // 2
        start_y = self.height - 100

        help_text = [
            "Tensor Universe Simulation Help",
            "",
            "Controls:",
            "- Press SPACE to return to menu",
            "- Press F to toggle full-screen mode",
            "- Press C to capture screenshot",
            "",
            "Physics Parameters:",
            "- Time Dilation: Shows gravitational time warping",
            "- Quantum Coherence: Measure of quantum state stability",
            "- Structures: Number of detected cosmic structures",
            "- Reality Density: Strength of reality projection from latent space",
            "",
            "Visual Settings:",
            "- Color Schemes: Different visualizations of density fields",
            "- Universe Size: Affects simulation scale and detail",
            "- Singularities: Number of gravitational centers",
            "",
            "Note: Higher numbers of singularities may affect performance"
        ]

        self.buttons.append(Button("Back to Main Menu", start_x, start_y, 
                                 button_width, button_height, 
                                 self.back_to_main, self.small_font))
        return help_text
    
    # Add restart method to Menu class
    def restart_simulation(self):
        """Restart the simulation with current parameters"""
        self.renderer.reinitialize_simulation()
        self.current_menu = "simulation"
        # Recreate simulation thread
        global sim_thread
        if sim_thread and sim_thread.is_alive():
            self.renderer.running = False
            sim_thread.join()
        self.renderer.running = True
        sim_thread = threading.Thread(target=simulation_thread_function, args=(self.renderer,))
        sim_thread.start()
    
    def setup_main_menu(self):
        """Initialize the main menu buttons"""
        self.buttons = []
        button_width = 300
        button_height = 50
        spacing = 20
        start_x = (self.width - button_width) // 2
        start_y = self.height // 2 - (button_height + spacing) * 2.5  # Adjusted for extra button

        # Main menu buttons
        menu_buttons = [
            ("Start Simulation", self.start_simulation),
            ("Restart Simulation", self.restart_simulation),
            ("Settings", self.open_settings),
            ("Help", self.open_help),
            ("Quit", self.quit)
        ]

        for i, (text, callback) in enumerate(menu_buttons):
            self.buttons.append(
                Button(text, 
                    start_x, 
                    start_y + i * (button_height + spacing),
                    button_width, 
                    button_height, 
                    callback, 
                    self.font)
            )


    def start_simulation(self):
        self.current_menu = "simulation"
        self.renderer.reinitialize_simulation()

    # Add these methods to the Menu class
    def increase_universe_size(self):
        self.renderer.universe_size += 10
        self.renderer.reinitialize_simulation()
        logging.info(f"Universe Size increased to {self.renderer.universe_size}")
        print(f"Universe Size increased to {self.renderer.universe_size}")
        self.setup_settings_menu()

    def decrease_universe_size(self):
        self.renderer.universe_size = max(10, self.renderer.universe_size - 10)
        self.renderer.reinitialize_simulation()
        logging.info(f"Universe Size decreased to {self.renderer.universe_size}")
        print(f"Universe Size decreased to {self.renderer.universe_size}")
        self.setup_settings_menu()

    def increase_singularities(self):
        self.renderer.num_singularities += 10
        self.renderer.reinitialize_simulation()
        logging.info(f"Number of Singularities increased to {self.renderer.num_singularities}")
        print(f"Number of Singularities increased to {self.renderer.num_singularities}")
        self.setup_settings_menu()

    def decrease_singularities(self):
        self.renderer.num_singularities = max(10, self.renderer.num_singularities - 10)
        self.renderer.reinitialize_simulation()
        logging.info(f"Number of Singularities decreased to {self.renderer.num_singularities}")
        print(f"Number of Singularities decreased to {self.renderer.num_singularities}")
        self.setup_settings_menu()

    def increase_steps(self):
        self.renderer.simulation_steps += 100
        logging.info(f"Simulation Steps increased to {self.renderer.simulation_steps}")
        print(f"Simulation Steps increased to {self.renderer.simulation_steps}")
        self.setup_settings_menu()

    def decrease_steps(self):
        self.renderer.simulation_steps = max(100, self.renderer.simulation_steps - 100)
        logging.info(f"Simulation Steps decreased to {self.renderer.simulation_steps}")
        print(f"Simulation Steps decreased to {self.renderer.simulation_steps}")
        self.setup_settings_menu()

    def change_color_scheme(self):
        current_index = self.color_schemes.index(self.selected_color_scheme)
        next_index = (current_index + 1) % len(self.color_schemes)
        self.selected_color_scheme = self.color_schemes[next_index]
        self.renderer.selected_color_scheme = self.selected_color_scheme
        logging.info(f"Color Scheme changed to {self.selected_color_scheme}")
        print(f"Color Scheme changed to {self.selected_color_scheme}")
        self.setup_settings_menu()

    def change_resolution(self):
        current_index = self.resolutions.index(self.selected_resolution)
        next_index = (current_index + 1) % len(self.resolutions)
        self.selected_resolution = self.resolutions[next_index]
        self.renderer.selected_resolution = self.selected_resolution
        self.screen = pygame.display.set_mode(self.selected_resolution, pygame.RESIZABLE)
        logging.info(f"Resolution changed to {self.selected_resolution}")
        print(f"Resolution changed to {self.selected_resolution}")
        self.setup_settings_menu()

    def open_settings(self):
        self.current_menu = "settings"
        self.setup_settings_menu()

    def open_help(self):
        self.current_menu = "help"
        self.setup_help_menu()

    def quit(self):
        pygame.quit()
        exit()

    def back_to_main(self):
        self.current_menu = "main"
        self.setup_main_menu()

    def setup_settings_menu(self):
        self.buttons = []
        button_width = 200
        button_height = 40
        spacing = 15
        start_x = self.width // 4
        start_y = 100

        settings = [
            ("Universe Size", self.increase_universe_size, self.decrease_universe_size),
            ("Singularities", self.increase_singularities, self.decrease_singularities),
            ("Simulation Steps", self.increase_steps, self.decrease_steps)
        ]

        for i, (name, inc_func, dec_func) in enumerate(settings):
            y_pos = start_y + i * 60
            self.buttons.extend([
                Button(f"Increase {name}", start_x, y_pos, 
                      button_width, button_height, inc_func, self.small_font),
                Button(f"Decrease {name}", start_x + 220, y_pos,
                      button_width, button_height, dec_func, self.small_font)
            ])

        # Add color scheme and resolution buttons
        self.buttons.extend([
            Button(f"Change Color Scheme ({self.selected_color_scheme})",
                  start_x, start_y + 180, button_width + 220, button_height,
                  self.change_color_scheme, self.small_font),
            Button(f"Change Resolution ({self.selected_resolution[0]}x{self.selected_resolution[1]})",
                  start_x, start_y + 240, button_width + 220, button_height,
                  self.change_resolution, self.small_font)
        ])

        self.buttons.append(
            Button("Back to Main Menu", start_x + 100, start_y + 300,
                  200, button_height, self.back_to_main, self.small_font)
        )

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Draw buttons
        for button in self.buttons:
            button.draw(self.screen)

        # Draw help text if in help menu
        if self.current_menu == "help":
            help_text = self.setup_help_menu()
            for i, line in enumerate(help_text):
                text_surf = self.small_font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surf, (50, 50 + i * 25))

        pygame.display.flip()


def main():
    global sim_thread
    sim_thread = None
    
    # Simulation parameters - reduce initial complexity
    universe_size = 50  # Reduced from 100
    num_singularities = 100  # Reduced from 200
    simulation_steps = 500  # Reduced from 1000
    
    # Initialize renderer
    renderer = SimulationRenderer(
        universe_size=universe_size,
        num_singularities=num_singularities,
        simulation_steps=simulation_steps,
        screen_size=(800, 600)  # Smaller initial window
    )
    
    # Initialize menu
    menu = Menu(renderer.screen, renderer)
    
    # Start simulation thread immediately
    sim_thread = threading.Thread(target=simulation_thread_function, args=(renderer,))
    sim_thread.daemon = True  # Make thread daemon so it exits with main program
    sim_thread.start()
    
    # Main loop
    try:
        while renderer.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    renderer.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        menu.current_menu = "main"
                        menu.setup_main_menu()
                    elif event.key == K_f:
                        renderer.toggle_fullscreen()
                    elif event.key == K_c:
                        renderer.capture_screenshot()
                elif event.type == MOUSEBUTTONDOWN:
                    if menu.current_menu != "simulation":
                        for button in menu.buttons:
                            if button.is_clicked(event.pos):
                                button.callback()
            
            if menu.current_menu != "simulation":
                menu.draw()
            else:
                renderer.draw()
            
            pygame.display.flip()
            renderer.clock.tick(30)  # Reduce to 30 FPS
            
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        renderer.running = False
        if sim_thread and sim_thread.is_alive():
            sim_thread.join(timeout=1.0)  # Wait max 1 second for thread to finish
        pygame.quit()

if __name__ == "__main__":
    main()
