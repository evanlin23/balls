import pygame
import os
import shutil
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip
import sys
import random
import math
import numpy as np
from scipy.io.wavfile import write

# --- Enhanced Configuration ---

# Video Settings
WIDTH, HEIGHT = 1080, 1920
FPS = 60
DURATION_SECONDS = 45

# Box and Ball Settings
BOX_PADDING = 50
BALL_RADIUS = 40
GRAVITY = 0.5
SIDE_BOUNCE_SPEED_INCREASE = 1.05
MIN_BOUNCE_INTERVAL = 3

# Enhanced Visual Settings
PARTICLE_COUNT = 35  # Increased particles
PARTICLE_LIFESPAN = 90  # Longer lasting particles
PARTICLE_MAX_VEL = 8
TRAIL_LENGTH = 25  # Longer trail
CAMERA_SHAKE_INTENSITY = 15
FLASH_DURATION = 8
GLOW_PULSE_SPEED = 0.1

# Enhanced Neon Colors with more variety
COLOR_BACKGROUND = (5, 0, 15)  # Darker purple background
COLOR_BOX = (0, 255, 255)  # Cyan
COLOR_LINE = (*COLOR_BOX, 120)
COLOR_BALL = (255, 0, 255)  # Magenta
COLOR_TEXT = (255, 255, 255, 150)
COLOR_PARTICLES = [(255, 100, 255), (100, 255, 255), (255, 255, 100), (255, 150, 0)]
COLOR_BEAT_FLASH = (255, 255, 255)

# File/Folder Names
FRAME_FOLDER = "temp_frames"
AUDIO_FOLDER = "sounds"
OUTPUT_VIDEO_FILE = "megalovania_ultimate_render.mp4"

# Enhanced Megalovania sequence with beat markers
MEGALOVANIA_NOTES = [
    293.66, 293.66, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    261.63, 261.63, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    246.94, 246.94, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
    233.08, 233.08, 587.33, 440.00, 415.30, 391.00, 349.23, 293.66, 349.23, 391.00,
]

# Beat emphasis points (every 4th note gets special treatment)
BEAT_EMPHASIS = [i for i in range(0, len(MEGALOVANIA_NOTES), 4)]

# --- Enhanced Audio Generation ---
def generate_enhanced_note_audio(frequency, duration=0.3, sample_rate=44100, is_beat=False):
    """Generate enhanced audio with better sound design"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create richer waveform
    wave = np.sin(2 * np.pi * frequency * t)
    wave += 0.4 * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
    wave += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)  # Third harmonic
    wave += 0.1 * np.sin(2 * np.pi * frequency * 4 * t)  # Fourth harmonic
    
    # Add some noise for texture
    noise = np.random.random(len(t)) * 0.02
    wave += noise
    
    # Enhanced envelope for beat notes
    fade_duration = 0.08 if is_beat else 0.05
    fade_samples = int(sample_rate * fade_duration)
    envelope = np.ones_like(wave)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Add vibrato for beat notes
    if is_beat:
        vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
        wave *= vibrato
        wave *= 1.3  # Louder for emphasis
    
    wave *= envelope
    
    # Normalize and convert
    wave = np.clip(wave * 0.35, -1.0, 1.0)
    audio = (wave * 32767).astype(np.int16)
    
    return audio

def create_megalovania_audio_files():
    """Create enhanced audio files"""
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
    
    print("Generating enhanced Megalovania audio...")
    for i, frequency in enumerate(MEGALOVANIA_NOTES):
        is_beat = i in BEAT_EMPHASIS
        audio_data = generate_enhanced_note_audio(frequency, is_beat=is_beat)
        filename = os.path.join(AUDIO_FOLDER, f"note_{i:03d}.wav")
        write(filename, 44100, audio_data)
    
    print(f"Generated {len(MEGALOVANIA_NOTES)} enhanced note files.")

# --- Enhanced Particle Class ---
class EnhancedParticle:
    def __init__(self, pos, color, is_beat=False):
        self.pos = list(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, PARTICLE_MAX_VEL * (1.5 if is_beat else 1))
        self.vel = [speed * math.cos(angle), speed * math.sin(angle)]
        self.color = random.choice(COLOR_PARTICLES) if is_beat else color
        self.lifespan = PARTICLE_LIFESPAN * (1.5 if is_beat else 1)
        self.max_lifespan = self.lifespan
        self.size = random.uniform(2, 6 if is_beat else 4)
        self.rotation = 0
        self.rotation_speed = random.uniform(-0.2, 0.2)

    def update(self):
        self.vel[1] += GRAVITY * 0.3
        self.vel[0] *= 0.98  # Air resistance
        self.vel[1] *= 0.98
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.rotation += self.rotation_speed

    def draw(self, surface):
        if self.lifespan <= 0:
            return
        
        alpha = max(0, 255 * (self.lifespan / self.max_lifespan))
        size = int(self.size * (self.lifespan / self.max_lifespan))
        
        # Create particle with glow effect
        particle_surf = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
        
        # Outer glow
        pygame.draw.circle(particle_surf, (*self.color, alpha // 3), 
                         (size * 3 // 2, size * 3 // 2), size * 2)
        # Inner bright core
        pygame.draw.circle(particle_surf, (*self.color, alpha), 
                         (size * 3 // 2, size * 3 // 2), size)
        
        surface.blit(particle_surf, (int(self.pos[0] - size * 1.5), int(self.pos[1] - size * 1.5)))

# --- Voronoi Background Effects ---
class VoronoiBackground:
    def __init__(self):
        self.seed_points = []
        self.colors = [
            (20, 0, 30),   # Dark purple
            (0, 20, 40),   # Dark blue
            (30, 0, 20),   # Dark red
            (0, 30, 30),   # Dark teal
            (25, 15, 0),   # Dark orange
        ]
        self.generate_initial_points()
    
    def generate_initial_points(self):
        """Generate initial Voronoi seed points"""
        for _ in range(3):  # Start with 3 points
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.seed_points.append([x, y])
    
    def add_bounce_point(self, bounce_pos):
        """Add a new Voronoi point when a bounce occurs"""
        # Add some randomness around the bounce point
        x = bounce_pos[0] + random.randint(-100, 100)
        y = bounce_pos[1] + random.randint(-100, 100)
        # Keep within bounds
        x = max(0, min(WIDTH, x))
        y = max(0, min(HEIGHT, y))
        self.seed_points.append([x, y])
        
        # Limit total points to prevent performance issues
        if len(self.seed_points) > 20:
            self.seed_points.pop(0)
    
    def draw_voronoi(self, surface):
        """Draw Voronoi diagram background"""
        if len(self.seed_points) < 2:
            return
        
        # Sample points across the screen and find nearest seed
        step = 8  # Lower = higher quality but slower
        for x in range(0, WIDTH, step):
            for y in range(0, HEIGHT, step):
                min_dist = float('inf')
                closest_seed = 0
                
                for i, seed in enumerate(self.seed_points):
                    dist = math.sqrt((x - seed[0])**2 + (y - seed[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_seed = i
                
                # Draw rectangle with color based on closest seed
                color_index = closest_seed % len(self.colors)
                color = self.colors[color_index]
                pygame.draw.rect(surface, color, (x, y, step, step))

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
        if self.shake_time <= 0:
            return (0, 0)
        
        shake_x = random.uniform(-self.shake_intensity, self.shake_intensity)
        shake_y = random.uniform(-self.shake_intensity, self.shake_intensity)
        return (int(shake_x), int(shake_y))

# --- Enhanced Frame Generation ---
def generate_enhanced_frames():
    if os.path.exists(FRAME_FOLDER):
        shutil.rmtree(FRAME_FOLDER)
    os.makedirs(FRAME_FOLDER)

    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 140)
    surface = pygame.Surface((WIDTH, HEIGHT))

    # Initialize enhanced objects
    box = pygame.Rect(BOX_PADDING, BOX_PADDING, WIDTH - 2 * BOX_PADDING, HEIGHT - 2 * BOX_PADDING)
    ball_pos = [WIDTH / 2, HEIGHT / 4]
    ball_vel = [7, 0]
    bounce_times, bounce_points, particles, trail_points = [], [], [], []
    bounce_multiplier = 1.0
    bounce_count = 0
    current_note_index = 0
    last_bounce_frame = -MIN_BOUNCE_INTERVAL
    
    # Enhanced effects
    voronoi_bg = VoronoiBackground()
    camera_shake = CameraShake()
    ball_glow_time = 0
    
    total_frames = DURATION_SECONDS * FPS
    print(f"Starting enhanced frame generation ({total_frames} frames)...")

    for frame_num in range(total_frames):
        if frame_num % 60 == 0:
            progress = (frame_num / total_frames) * 100
            sys.stdout.write(f"\rGenerating enhanced frames: {progress:.1f}% complete")
            sys.stdout.flush()
        
        # --- Enhanced Physics ---
        ball_vel[1] += GRAVITY
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        
        trail_points.append(list(ball_pos))
        if len(trail_points) > TRAIL_LENGTH:
            trail_points.pop(0)

        # --- Enhanced Collision Detection ---
        bounced, contact_point = False, None
        can_bounce = (frame_num - last_bounce_frame) >= MIN_BOUNCE_INTERVAL
        is_beat_bounce = False
        
        # Collision logic (same as original but with beat detection)
        if ball_pos[0] - BALL_RADIUS <= box.left and ball_vel[0] < 0 and can_bounce:
            ball_pos[0] = box.left + BALL_RADIUS + 2
            ball_vel[0] = abs(ball_vel[0]) * SIDE_BOUNCE_SPEED_INCREASE
            ball_vel[1] *= SIDE_BOUNCE_SPEED_INCREASE
            bounced = True
            contact_point = (box.left, ball_pos[1])
            
        elif ball_pos[0] + BALL_RADIUS >= box.right and ball_vel[0] > 0 and can_bounce:
            ball_pos[0] = box.right - BALL_RADIUS - 2
            ball_vel[0] = -abs(ball_vel[0]) * SIDE_BOUNCE_SPEED_INCREASE
            ball_vel[1] *= SIDE_BOUNCE_SPEED_INCREASE
            bounced = True
            contact_point = (box.right, ball_pos[1])
            
        if ball_pos[1] - BALL_RADIUS <= box.top and ball_vel[1] < 0 and can_bounce and not bounced:
            ball_pos[1] = box.top + BALL_RADIUS + 2
            ball_vel[1] = abs(ball_vel[1])
            bounced = True
            contact_point = (ball_pos[0], box.top)
            
        elif ball_pos[1] + BALL_RADIUS >= box.bottom and ball_vel[1] > 0 and can_bounce and not bounced:
            ball_pos[1] = box.bottom - BALL_RADIUS - 2
            ball_vel[1] = -abs(ball_vel[1])
            bounced = True
            contact_point = (ball_pos[0], box.bottom)
        
        if bounced:
            bounce_times.append(frame_num / FPS)
            bounce_multiplier *= SIDE_BOUNCE_SPEED_INCREASE
            bounce_count += 1
            current_note_index = (bounce_count - 1) % len(MEGALOVANIA_NOTES)
            is_beat_bounce = current_note_index in BEAT_EMPHASIS
            last_bounce_frame = frame_num
            bounce_points.append(contact_point)
            
            # Enhanced effects on bounce
            camera_shake.trigger(CAMERA_SHAKE_INTENSITY * (2 if is_beat_bounce else 1))
            ball_glow_time = 15
            
            # Add bounce point to Voronoi background
            if contact_point:
                voronoi_bg.add_bounce_point(contact_point)
                
            if contact_point:
                particle_count = PARTICLE_COUNT * (2 if is_beat_bounce else 1)
                for _ in range(particle_count):
                    particles.append(EnhancedParticle(contact_point, COLOR_BALL, is_beat_bounce))

        # --- Update Effects ---
        camera_shake.update()
        if ball_glow_time > 0:
            ball_glow_time -= 1

        # --- Enhanced Drawing ---
        surface.fill(COLOR_BACKGROUND)
        
        # Draw Voronoi background
        voronoi_bg.draw_voronoi(surface)
        
        # Get camera shake offset
        shake_offset = camera_shake.get_offset()
        
        # Create main drawing surface with shake
        draw_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # Draw enhanced particles
        for p in particles:
            p.update()
            p.draw(draw_surface)
        particles = [p for p in particles if p.lifespan > 0]

        # Draw single-color trail
        for i, point in enumerate(trail_points):
            trail_alpha = (i / TRAIL_LENGTH) * 150
            trail_radius = (i / TRAIL_LENGTH) * BALL_RADIUS * 0.9
                
            pygame.draw.circle(draw_surface, (*COLOR_BALL, int(trail_alpha)), 
                             [int(point[0]), int(point[1])], int(trail_radius))

        # Draw enhanced tether lines with pulsing
        line_alpha = 80 + int(40 * math.sin(frame_num * 0.1))
        for i, point in enumerate(bounce_points):
            line_color = (*COLOR_LINE[:3], line_alpha)
            pygame.draw.line(draw_surface, line_color, point, 
                           (int(ball_pos[0]), int(ball_pos[1])), 3)

        # Draw enhanced multiplier with scaling effect
        scale_factor = 1.0 + (ball_glow_time / 15) * 0.3
        scaled_font_size = int(140 * scale_factor)
        scaled_font = pygame.font.Font(None, scaled_font_size)
        multiplier_text = scaled_font.render(f"{bounce_multiplier:.2f}x", True, COLOR_TEXT)
        text_rect = multiplier_text.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        draw_surface.blit(multiplier_text, text_rect)

        # Draw enhanced box with pulsing border
        box_thickness = 5 + int(3 * math.sin(frame_num * 0.1))
        pygame.draw.rect(draw_surface, COLOR_BOX, box, box_thickness, border_radius=15)

        # Draw enhanced ball with dynamic glow
        ball_glow_intensity = 1.0 + (ball_glow_time / 15) * 1.5
        glow_radius = int(BALL_RADIUS * 2 * ball_glow_intensity)
        
        # Multiple glow layers for depth
        for layer in range(3):
            layer_radius = glow_radius - (layer * 15)
            layer_alpha = 30 - (layer * 10)
            if layer_radius > 0:
                glow_surface = pygame.Surface((layer_radius * 2, layer_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*COLOR_BALL, layer_alpha), 
                                 (layer_radius, layer_radius), layer_radius)
                draw_surface.blit(glow_surface, (int(ball_pos[0]) - layer_radius, 
                                               int(ball_pos[1]) - layer_radius))

        # Main ball
        pygame.draw.circle(draw_surface, COLOR_BALL, 
                         (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)

        # Apply camera shake and draw to main surface
        surface.blit(draw_surface, shake_offset)

        # Save frame
        pygame.image.save(surface, os.path.join(FRAME_FOLDER, f"frame_{frame_num:05d}.png"))
        
    pygame.quit()
    sys.stdout.write("\rEnhanced frame generation complete!         \n")
    print(f"Total bounces: {len(bounce_times)}")
    return bounce_times

# --- Enhanced Video Compilation ---
def create_enhanced_video(bounce_times):
    print("\nCompiling enhanced video with audio...")
    frame_files = sorted([os.path.join(FRAME_FOLDER, f) for f in os.listdir(FRAME_FOLDER) if f.endswith(".png")])
    video_clip = ImageSequenceClip(frame_files, fps=FPS)

    if not bounce_times:
        print("WARNING: No bounces recorded. Creating silent video.")
        final_clip = video_clip
    else:
        try:
            audio_clips = []
            for i, bounce_time in enumerate(bounce_times):
                if bounce_time < video_clip.duration:
                    note_index = i % len(MEGALOVANIA_NOTES)
                    note_file = os.path.join(AUDIO_FOLDER, f"note_{note_index:03d}.wav")
                    
                    if os.path.exists(note_file):
                        note_audio = AudioFileClip(note_file)
                        audio_clips.append(note_audio.set_start(bounce_time))
            
            if audio_clips:
                composite_audio = CompositeAudioClip(audio_clips).set_duration(video_clip.duration)
                final_clip = video_clip.set_audio(composite_audio)
                print("Enhanced Megalovania audio added successfully!")
            else:
                final_clip = video_clip
        except Exception as e:
            print(f"Audio error: {e}\nCreating video without audio...")
            final_clip = video_clip

    print("Writing final enhanced MP4...")
    final_clip.write_videofile(
        OUTPUT_VIDEO_FILE, codec="libx264", audio_codec="aac", 
        audio_bitrate="192k", preset="medium", threads=4, 
        verbose=False, logger=None
    )
    
    shutil.rmtree(FRAME_FOLDER)
    print(f"Enhanced video saved as {OUTPUT_VIDEO_FILE}!")

# --- Run Enhanced Script ---
if __name__ == '__main__':
    create_megalovania_audio_files()
    bounce_timestamps = generate_enhanced_frames()
    create_enhanced_video(bounce_timestamps)