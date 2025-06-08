import pygame
import os
import shutil
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip
import sys
import random
import math
import numpy as np
from scipy.io.wavfile import write
import pygame.surfarray as surfarray

# --- Enhanced Configuration ---

# Video Settings
WIDTH, HEIGHT = 1080, 1920
FPS = 60
DURATION_SECONDS = 15 # Note: Total runtime will be DURATION_SECONDS * 6

# Box and Ball Settings
BOX_PADDING = 100
BALL_RADIUS = 40
GRAVITY = 0.5
SIDE_BOUNCE_SPEED_INCREASE = 1.05
MIN_BOUNCE_INTERVAL = 3

# Enhanced Visual Settings
PARTICLE_COUNT = 35
PARTICLE_LIFESPAN = 90
PARTICLE_MAX_VEL = 8
TRAIL_LENGTH = 25
CAMERA_SHAKE_INTENSITY = 15
FLASH_DURATION = 8
GLOW_PULSE_SPEED = 0.1

# Enhanced Neon Colors with more variety
COLOR_BACKGROUND_IN_BOX = (5, 0, 15)  # Darker purple background inside the box
COLOR_BOX = (0, 255, 255)  # Cyan
COLOR_LINE = (*COLOR_BOX, 120)
COLOR_BALL = (255, 0, 255)  # Magenta
COLOR_TEXT = (255, 255, 255, 150)
COLOR_PARTICLES = [(255, 100, 255), (100, 255, 255), (255, 255, 100), (255, 150, 0)]
COLOR_BEAT_FLASH = (255, 255, 255)

# File/Folder Names
FRAME_FOLDER = "temp_frames"
AUDIO_FOLDER = "sounds"

# Enhanced Megalovania sequence
MEGALOVANIA_NOTES = [
    293.66, 293.66, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    261.63, 261.63, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    246.94, 246.94, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    233.08, 233.08, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
]

# --- Enhanced Audio Generation ---
def generate_enhanced_note_audio(frequency, duration=0.3, sample_rate=44100, is_beat=False):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) + 0.4 * np.sin(2 * np.pi * frequency * 2 * t) + 0.2 * np.sin(2 * np.pi * frequency * 3 * t) + 0.1 * np.sin(2 * np.pi * frequency * 4 * t)
    noise = np.random.random(len(t)) * 0.02
    wave += noise
    fade_duration = 0.08 if is_beat else 0.05
    fade_samples = int(sample_rate * fade_duration)
    envelope = np.ones_like(wave)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    if is_beat:
        vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
        wave *= vibrato
        wave *= 1.3
    wave *= envelope
    wave = np.clip(wave * 0.35, -1.0, 1.0)
    audio = (wave * 32767).astype(np.int16)
    return audio

def create_megalovania_audio_files():
    if os.path.exists(AUDIO_FOLDER) and len(os.listdir(AUDIO_FOLDER)) == len(MEGALOVANIA_NOTES):
        print("Megalovania audio files already exist. Skipping generation.")
        return
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
    print("Generating enhanced Megalovania audio...")
    for i, frequency in enumerate(MEGALOVANIA_NOTES):
        # Every note is now generated as an emphasized 'beat' note.
        is_beat = True
        audio_data = generate_enhanced_note_audio(frequency, is_beat=is_beat)
        filename = os.path.join(AUDIO_FOLDER, f"note_{i:03d}.wav")
        write(filename, 44100, audio_data)
    print(f"Generated {len(MEGALOVANIA_NOTES)} enhanced note files.")

# --- Enhanced Particle Class ---
class EnhancedParticle:
    def __init__(self, pos, color, is_beat=False):
        self.pos = list(pos)
        angle = random.uniform(0, 2 * math.pi)
        # All particles now get the emphasized speed, lifespan, and size
        speed = random.uniform(3, PARTICLE_MAX_VEL * (1.5 if is_beat else 1))
        self.vel = [speed * math.cos(angle), speed * math.sin(angle)]
        self.color = random.choice(COLOR_PARTICLES)
        self.lifespan = PARTICLE_LIFESPAN * (1.5 if is_beat else 1)
        self.max_lifespan = self.lifespan
        self.size = random.uniform(2, 6 if is_beat else 4)

    def update(self):
        self.vel[1] += GRAVITY * 0.3
        self.vel[0] *= 0.98
        self.vel[1] *= 0.98
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan <= 0: return
        alpha = max(0, 255 * (self.lifespan / self.max_lifespan))
        size = int(self.size * (self.lifespan / self.max_lifespan))
        particle_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, (*self.color, alpha // 3), (size * 2, size * 2), size * 2)
        pygame.draw.circle(particle_surf, (*self.color, alpha), (size * 2, size * 2), size)
        surface.blit(particle_surf, (int(self.pos[0] - size * 2), int(self.pos[1] - size * 2)))

# --- Camera Shake ---
class CameraShake:
    def __init__(self):
        self.shake_time = 0
        self.shake_intensity = 0
    def trigger(self, intensity=CAMERA_SHAKE_INTENSITY):
        self.shake_time = 10
        self.shake_intensity = intensity
    def update(self):
        if self.shake_time > 0:
            self.shake_time -= 1
            self.shake_intensity *= 0.9
    def get_offset(self):
        if self.shake_time <= 0: return (0, 0)
        return (int(random.uniform(-self.shake_intensity, self.shake_intensity)), int(random.uniform(-self.shake_intensity, self.shake_intensity)))

# --- All Background Effect Classes ---

# Option 0: Voronoi Background
class VoronoiBackground:
    def __init__(self):
        self.seed_points = []
        self.colors = [(20, 0, 30), (0, 20, 40), (30, 0, 20), (0, 30, 30), (25, 15, 0)]
        self.generate_initial_points()
    def generate_initial_points(self):
        for _ in range(3):
            self.seed_points.append([random.randint(0, WIDTH), random.randint(0, HEIGHT)])
    def add_bounce_point(self, bounce_pos):
        x = max(0, min(WIDTH, bounce_pos[0] + random.randint(-100, 100)))
        y = max(0, min(HEIGHT, bounce_pos[1] + random.randint(-100, 100)))
        self.seed_points.append([x, y])
        if len(self.seed_points) > 20: self.seed_points.pop(0)
    def draw_voronoi(self, surface):
        if len(self.seed_points) < 2: return
        step = 8
        for x in range(0, WIDTH, step):
            for y in range(0, HEIGHT, step):
                min_dist = float('inf')
                closest_seed = 0
                for i, seed in enumerate(self.seed_points):
                    dist = math.sqrt((x - seed[0])**2 + (y - seed[1])**2)
                    if dist < min_dist:
                        min_dist, closest_seed = dist, i
                color = self.colors[closest_seed % len(self.colors)]
                pygame.draw.rect(surface, color, (x, y, step, step))
    def update(self, frame_num, bounced): # Dummy update for compatibility
        pass

# Option 1: Pulsing Grid Background (IMPROVED)
class PulsingGrid:
    def __init__(self, width, height):
        self.width, self.height, self.grid_size = width, height, 80
        self.pulse_time = 0
        self.offset_x, self.offset_y = 0.0, 0.0
        self.warp_amp, self.warp_freq = 0.0, 0.02
        self.color = (0, 100, 150)

    def update(self, frame_num, bounce_occurred=False):
        self.pulse_time = frame_num
        # Constant scrolling
        self.offset_x = (self.offset_x + 0.5) % self.grid_size
        self.offset_y = (self.offset_y + 0.3) % self.grid_size
        # Trigger warp on bounce
        if bounce_occurred:
            self.warp_amp = 20.0
        # Decay warp effect
        self.warp_amp *= 0.95
        if self.warp_amp < 0.1: self.warp_amp = 0.0

    def draw(self, surface):
        pulse_intensity = 0.6 + 0.4 * math.sin(self.pulse_time * 0.1)
        line_alpha = int(60 * pulse_intensity)
        if line_alpha < 5: return
        
        line_color = (*self.color, line_alpha)
        
        # Draw warped vertical lines
        for x_base in range(int(-self.offset_x), self.width, self.grid_size):
            points = []
            for y in range(0, self.height + 10, 10): # Segmented for warping
                x_offset = math.sin(y * self.warp_freq + self.pulse_time * 0.05) * self.warp_amp
                points.append((x_base + x_offset, y))
            if len(points) > 1:
                pygame.draw.aalines(surface, line_color, False, points, 1)

        # Draw warped horizontal lines
        for y_base in range(int(-self.offset_y), self.height, self.grid_size):
            points = []
            for x in range(0, self.width + 10, 10):
                y_offset = math.sin(x * self.warp_freq + self.pulse_time * 0.05) * self.warp_amp
                points.append((x, y_base + y_offset))
            if len(points) > 1:
                pygame.draw.aalines(surface, line_color, False, points, 1)

# Option 2: Ripple Wave Background (IMPROVED)
class RippleWaves:
    def __init__(self, width, height):
        self.width, self.height, self.ripples = width, height, []
        # A ripple is definitely off-screen when its radius exceeds the diagonal
        self.max_ripple_radius = math.hypot(width, height)

    def add_ripple(self, pos, intensity=1.0, is_beat=False):
        # [x, y, age, intensity, is_beat]
        self.ripples.append([pos[0], pos[1], 0, intensity, is_beat])

    def update(self):
        for ripple in self.ripples[:]:
            ripple[2] += 1 # age
            speed = 6 if ripple[4] else 4
            radius = ripple[2] * speed
            # Remove ripple only when it's guaranteed to be off-screen
            if radius > self.max_ripple_radius + 100:
                self.ripples.remove(ripple)

    def draw(self, surface):
        for x, y, time, intensity, is_beat in self.ripples:
            speed = 6 if is_beat else 4
            radius = time * speed
            
            # Fade out based on size relative to the screen, not a fixed lifespan
            fade_ratio = min(1.0, radius / self.max_ripple_radius)
            alpha = (1 - fade_ratio**2) * 200 * intensity
            
            if alpha <= 5: continue
            
            color = (int(255 - 200 * fade_ratio), 255, 255)
            glow_width = 15 if is_beat else 10
            line_width = 4 if is_beat else 2
            
            if radius > 0:
                pygame.draw.circle(surface, (*color, int(alpha * 0.3)), (int(x), int(y)), int(radius), width=glow_width)
                pygame.draw.circle(surface, (255, 255, 255, int(alpha * 0.8)), (int(x), int(y)), int(radius), width=line_width)

# Option 3: Floating Geometric Shapes
class FloatingShapes:
    def __init__(self, width, height):
        self.width, self.height, self.shapes = width, height, []
        self.shape_types = ['triangle', 'diamond', 'hexagon']
        self.colors = [(50, 255, 255), (255, 50, 255), (255, 255, 50), (100, 255, 100)]
        for _ in range(15): self.add_random_shape()
    def add_random_shape(self):
        self.shapes.append({'type': random.choice(self.shape_types), 'pos': [random.uniform(0, self.width), random.uniform(0, self.height)], 'vel': [random.uniform(-1, 1), random.uniform(-1, 1)], 'size': random.uniform(20, 60), 'rotation': 0, 'rotation_speed': random.uniform(-0.02, 0.02), 'color': random.choice(self.colors), 'alpha': random.uniform(30, 80)})
    def update(self, bounce_occurred=False):
        for shape in self.shapes:
            shape['pos'][0] += shape['vel'][0]
            shape['pos'][1] += shape['vel'][1]
            shape['rotation'] += shape['rotation_speed']
            if shape['pos'][0] < -shape['size']: shape['pos'][0] = self.width + shape['size']
            elif shape['pos'][0] > self.width + shape['size']: shape['pos'][0] = -shape['size']
            if shape['pos'][1] < -shape['size']: shape['pos'][1] = self.height + shape['size']
            elif shape['pos'][1] > self.height + shape['size']: shape['pos'][1] = -shape['size']
            if bounce_occurred: shape['alpha'] = min(150, shape['alpha'] * 1.5)
            else: shape['alpha'] *= 0.995
    def draw(self, surface):
        for shape in self.shapes: self.draw_shape(surface, shape)
    def draw_shape(self, surface, shape):
        x, y, size, color = shape['pos'][0], shape['pos'][1], shape['size'], (*shape['color'], int(shape['alpha']))
        points = []
        if shape['type'] == 'triangle': points = [(x, y - size), (x - size * 0.866, y + size * 0.5), (x + size * 0.866, y + size * 0.5)]
        elif shape['type'] == 'diamond': points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        elif shape['type'] == 'hexagon':
            for i in range(6): points.append((x + size * math.cos(i * math.pi / 3 + shape['rotation']), y + size * math.sin(i * math.pi / 3 + shape['rotation'])))
        if shape['rotation'] != 0 and shape['type'] != 'hexagon':
            cos_r, sin_r, rotated_points = math.cos(shape['rotation']), math.sin(shape['rotation']), []
            for px, py in points: rotated_points.append(((px - x) * cos_r - (py - y) * sin_r + x, (px - x) * sin_r + (py - y) * cos_r + y))
            points = rotated_points
        if len(points) > 2: pygame.draw.polygon(surface, color, points, 2)

# Option 4: Particle Field Background
class ParticleField:
    def __init__(self, width, height):
        self.width, self.height, self.particles = width, height, []
        for _ in range(100): self.particles.append({'pos': [random.uniform(0, width), random.uniform(0, height)], 'vel': [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)], 'size': random.uniform(1, 3), 'brightness': random.uniform(0.3, 1.0), 'pulse_offset': random.uniform(0, 2 * math.pi)})
    def update(self, frame_num, ball_pos=None, bounce_occurred=False):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if p['pos'][0] < 0: p['pos'][0] = self.width
            elif p['pos'][0] > self.width: p['pos'][0] = 0
            if p['pos'][1] < 0: p['pos'][1] = self.height
            elif p['pos'][1] > self.height: p['pos'][1] = 0
            if ball_pos:
                dx, dy = ball_pos[0] - p['pos'][0], ball_pos[1] - p['pos'][1]
                dist_sq = dx*dx + dy*dy
                if 0 < dist_sq < 200**2:
                    force = 200 / (dist_sq + 1)
                    p['vel'][0] += dx * force * 0.01
                    p['vel'][1] += dy * force * 0.01
                    speed = math.sqrt(p['vel'][0]**2 + p['vel'][1]**2)
                    if speed > 2: p['vel'][0], p['vel'][1] = (p['vel'][0] / speed) * 2, (p['vel'][1] / speed) * 2
    def draw(self, surface, frame_num):
        for p in self.particles:
            pulse = 0.5 + 0.5 * math.sin(frame_num * 0.05 + p['pulse_offset'])
            alpha = int(p['brightness'] * pulse * 100)
            pygame.draw.circle(surface, (200, 200, 255, alpha), (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

# Option 5: Sine Wave Interference Pattern (IMPROVED)
class SineWaveBackground:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.wave_sources = [{'pos': (width * 0.25, height * 0.25), 'frequency': 0.02, 'phase': 0}, {'pos': (width * 0.75, height * 0.75), 'frequency': 0.025, 'phase': math.pi}, {'pos': (width * 0.5, height * 0.1), 'frequency': 0.018, 'phase': math.pi/2}]
        # Pre-calculate coordinate grid for performance
        self.xx, self.yy = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    def add_wave_source(self, pos):
        self.wave_sources.append({'pos': pos, 'frequency': random.uniform(0.015, 0.03), 'phase': random.uniform(0, 2 * math.pi), 'amplitude': 1.0, 'decay': 0.995})
        if len(self.wave_sources) > 8: self.wave_sources.pop(0)

    def update(self, frame_num):
        for s in self.wave_sources:
            s['phase'] += s['frequency']
            if 'decay' in s: s['amplitude'] *= s['decay']
        self.wave_sources = [s for s in self.wave_sources if s.get('amplitude', 1.0) > 0.1]

    def draw(self, surface):
        if not self.wave_sources: return
        
        # Using NumPy for smooth, fast, per-pixel calculation
        wave_sum_array = np.zeros((self.width, self.height), dtype=np.float32)

        for s in self.wave_sources:
            pos_x, pos_y = s['pos']
            dist = np.sqrt((self.xx - pos_x)**2 + (self.yy - pos_y)**2)
            # Add a small epsilon to distance to avoid division by zero at the source center
            wave_sum_array += s.get('amplitude', 1.0) * np.sin(dist * 0.05 + s['phase']) / (dist * 0.01 + 1)
        
        # Normalize the summed waves to an intensity value between 0 and 1
        intensity = np.clip((wave_sum_array + 1) * 0.5, 0, 1)
        
        # Create an RGB pixel array from the intensity map
        pixel_array = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        pixel_array[:, :, 1] = (intensity * 100).astype(np.uint8) # Green channel
        pixel_array[:, :, 2] = (intensity * 200).astype(np.uint8) # Blue channel
        
        # Blit the NumPy array directly to the surface for a perfectly smooth render
        surfarray.blit_array(surface, pixel_array)

# --- Dictionary of all available background types ---
BACKGROUND_TYPES = {
    "Voronoi": VoronoiBackground,
    "PulsingGrid": PulsingGrid,
    "RippleWaves": RippleWaves,
    "FloatingShapes": FloatingShapes,
    "ParticleField": ParticleField,
    "SineWaveBackground": SineWaveBackground,
}

# --- Enhanced Frame Generation (Now with contained backgrounds) ---
def generate_enhanced_frames(background_class, background_name):
    if os.path.exists(FRAME_FOLDER):
        shutil.rmtree(FRAME_FOLDER)
    os.makedirs(FRAME_FOLDER)

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((WIDTH, HEIGHT))

    box = pygame.Rect(BOX_PADDING, BOX_PADDING, WIDTH - 2 * BOX_PADDING, HEIGHT - 2 * BOX_PADDING)
    ball_pos, ball_vel = [WIDTH / 2, HEIGHT / 4], [7, 0]
    bounce_times, bounce_points, particles, trail_points = [], [], [], []
    bounce_multiplier, bounce_count = 1.0, 0
    last_bounce_frame = -MIN_BOUNCE_INTERVAL
    camera_shake, ball_glow_time = CameraShake(), 0
    
    background_effect = background_class(WIDTH, HEIGHT) if background_name != "Voronoi" else background_class()

    total_frames = DURATION_SECONDS * FPS
    print(f"Starting frame generation ({total_frames} frames) for '{background_name}'...")

    for frame_num in range(total_frames):
        if frame_num % 60 == 0:
            sys.stdout.write(f"\rGenerating frames: {(frame_num / total_frames) * 100:.1f}% complete")
            sys.stdout.flush()
        
        # Physics
        ball_vel[1] += GRAVITY
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        trail_points.append(list(ball_pos))
        if len(trail_points) > TRAIL_LENGTH: trail_points.pop(0)

        # Collision
        bounced, contact_point = False, None
        if (frame_num - last_bounce_frame) >= MIN_BOUNCE_INTERVAL:
            if ball_pos[0] - BALL_RADIUS <= box.left and ball_vel[0] < 0:
                ball_pos[0], ball_vel[0] = box.left + BALL_RADIUS + 2, abs(ball_vel[0]) * SIDE_BOUNCE_SPEED_INCREASE; ball_vel[1] *= SIDE_BOUNCE_SPEED_INCREASE; bounced, contact_point = True, (box.left, ball_pos[1])
            elif ball_pos[0] + BALL_RADIUS >= box.right and ball_vel[0] > 0:
                ball_pos[0], ball_vel[0] = box.right - BALL_RADIUS - 2, -abs(ball_vel[0]) * SIDE_BOUNCE_SPEED_INCREASE; ball_vel[1] *= SIDE_BOUNCE_SPEED_INCREASE; bounced, contact_point = True, (box.right, ball_pos[1])
            elif ball_pos[1] - BALL_RADIUS <= box.top and ball_vel[1] < 0:
                ball_pos[1], ball_vel[1] = box.top + BALL_RADIUS + 2, abs(ball_vel[1]); bounced, contact_point = True, (ball_pos[0], box.top)
            elif ball_pos[1] + BALL_RADIUS >= box.bottom and ball_vel[1] > 0:
                ball_pos[1], ball_vel[1] = box.bottom - BALL_RADIUS - 2, -abs(ball_vel[1]); bounced, contact_point = True, (ball_pos[0], box.bottom)
        
        if bounced:
            bounce_times.append(frame_num / FPS)
            bounce_multiplier *= SIDE_BOUNCE_SPEED_INCREASE
            bounce_count += 1
            # --- MODIFIED --- Force is_beat_bounce to True for every bounce
            is_beat_bounce = True
            last_bounce_frame = frame_num
            bounce_points.append(contact_point)
            # Camera shake intensity is now always the maximum amount.
            camera_shake.trigger(CAMERA_SHAKE_INTENSITY * 2)
            ball_glow_time = 15
            if contact_point:
                # Particle count is now always the maximum amount.
                for _ in range(PARTICLE_COUNT * 2):
                    particles.append(EnhancedParticle(contact_point, COLOR_BALL, is_beat_bounce))

        # --- DYNAMIC BACKGROUND UPDATE ---
        if bounced and contact_point:
            # is_beat_bounce is now always True, so background effects will use their emphasized states
            if background_name == "RippleWaves":
                background_effect.add_ripple(contact_point, intensity=1.5, is_beat=is_beat_bounce)
            elif background_name == "SineWaveBackground":
                background_effect.add_wave_source(contact_point)
            elif background_name == "Voronoi":
                background_effect.add_bounce_point(contact_point)
        
        # All background updates now happen every frame
        if background_name == "RippleWaves":
            background_effect.update()
        elif background_name == "SineWaveBackground":
            background_effect.update(frame_num)
        elif background_name == "FloatingShapes":
            background_effect.update(bounce_occurred=bounced)
        elif background_name == "ParticleField":
            background_effect.update(frame_num, ball_pos, bounced)
        else: # This now correctly handles PulsingGrid and Voronoi
            background_effect.update(frame_num, bounced)

        camera_shake.update()
        if ball_glow_time > 0: ball_glow_time -= 1

        # Drawing with Contained Background
        surface.fill((0, 0, 0))
        surface.set_clip(box)
        
        surface.fill(COLOR_BACKGROUND_IN_BOX)
        if background_name == "Voronoi": background_effect.draw_voronoi(surface)
        elif background_name == "ParticleField": background_effect.draw(surface, frame_num)
        else: background_effect.draw(surface)
        
        surface.set_clip(None)
        
        draw_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for p in particles:
            p.update(); p.draw(draw_surface)
        particles = [p for p in particles if p.lifespan > 0]
        
        for i, point in enumerate(trail_points):
            pygame.draw.circle(draw_surface, (*COLOR_BALL, int((i / TRAIL_LENGTH) * 150)), [int(p) for p in point], int((i / TRAIL_LENGTH) * BALL_RADIUS * 0.9))
        for point in bounce_points:
            pygame.draw.line(draw_surface, (*COLOR_LINE[:3], 80 + int(40 * math.sin(frame_num * 0.1))), point, (int(ball_pos[0]), int(ball_pos[1])), 3)
        
        scale_factor = 1.0 + (ball_glow_time / 15) * 0.3
        multiplier_text = pygame.font.Font(None, int(140 * scale_factor)).render(f"{bounce_multiplier:.2f}x", True, COLOR_TEXT)
        draw_surface.blit(multiplier_text, multiplier_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
        pygame.draw.rect(draw_surface, COLOR_BOX, box, 5 + int(3 * math.sin(frame_num * 0.1)), border_radius=15)

        glow_radius = int(BALL_RADIUS * 2 * (1.0 + (ball_glow_time / 15) * 1.5))
        for layer in range(3):
            if (r := glow_radius - (layer * 15)) > 0:
                glow_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA); pygame.draw.circle(glow_surface, (*COLOR_BALL, 30 - (layer * 10)), (r, r), r); draw_surface.blit(glow_surface, (int(ball_pos[0]) - r, int(ball_pos[1]) - r))
        
        pygame.draw.circle(draw_surface, COLOR_BALL, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)
        
        surface.blit(draw_surface, camera_shake.get_offset())
        pygame.image.save(surface, os.path.join(FRAME_FOLDER, f"frame_{frame_num:05d}.png"))
        
    pygame.quit()
    sys.stdout.write("\rFrame generation complete!                  \n")
    return bounce_times

# --- Enhanced Video Compilation ---
def create_enhanced_video(bounce_times, output_filename):
    print(f"\nCompiling video with audio to '{output_filename}'...")
    frame_files = sorted([os.path.join(FRAME_FOLDER, f) for f in os.listdir(FRAME_FOLDER) if f.endswith(".png")])
    video_clip = ImageSequenceClip(frame_files, fps=FPS)

    if not bounce_times:
        final_clip = video_clip
    else:
        try:
            audio_clips = [AudioFileClip(os.path.join(AUDIO_FOLDER, f"note_{i % len(MEGALOVANIA_NOTES):03d}.wav")).set_start(t) for i, t in enumerate(bounce_times) if t < video_clip.duration and os.path.exists(os.path.join(AUDIO_FOLDER, f"note_{i % len(MEGALOVANIA_NOTES):03d}.wav"))]
            final_clip = video_clip.set_audio(CompositeAudioClip(audio_clips).set_duration(video_clip.duration)) if audio_clips else video_clip
        except Exception as e:
            print(f"Audio error: {e}. Creating video without audio.")
            final_clip = video_clip

    final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", audio_bitrate="192k", preset="medium", threads=4, verbose=False, logger=None)
    print(f"Video saved as {output_filename}!")

# --- Main Script Runner ---
if __name__ == '__main__':
    create_megalovania_audio_files()

    num_backgrounds = len(BACKGROUND_TYPES)
    for i, (name, bg_class) in enumerate(BACKGROUND_TYPES.items()):
        print(f"\n{'='*20} Video {i+1}/{num_backgrounds}: {name} {'='*20}")
        bounce_timestamps = generate_enhanced_frames(bg_class, name)
        output_file = f"video_{name.lower().replace(' ', '')}.mp4"
        create_enhanced_video(bounce_timestamps, output_file)
        
    if os.path.exists(FRAME_FOLDER):
        shutil.rmtree(FRAME_FOLDER)

    print(f"\n{'='*20} ALL {num_backgrounds} VIDEOS CREATED SUCCESSFULLY {'='*20}")