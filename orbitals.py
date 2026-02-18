import pygame
import sys
import math
import random
import numpy as np
from scipy.special import genlaguerre

# ------------------------- CONFIGURATION -------------------------
# Initial quantum numbers (n, l, m)
n = 3
l = 2
m = 0

brightness_scale = 50.0       # overall brightness scaling
num_points = 10000            # number of particles
rmax = 50.0                   # maximum radial extent (Bohr units)
max_drift = 2.0               # maximum drift from anchor (Bohr units)
movement_strength = 0.1       # initial speed (Bohr units/frame)
random_accel_strength = 0.02  # random acceleration per frame
max_speed = 0.05              # speed cap (Bohr units/frame)
transition_speed = 0.5        # speed at which anchors transition (Bohr units/frame)

# NEW: Cursor repulsion settings
cursor_repulsion_distance = 1.1    # distance within which particles are repelled (simulation units)
cursor_repulsion_strength = 1    # strength of repulsion acceleration

print(f"Initial: Hydrogen Orbital (n={n}, l={l}, m={m})")
print(f"Movement strength={movement_strength}, max_drift={max_drift}, random accel={random_accel_strength}, max_speed={max_speed}, transition_speed={transition_speed}")
print(f"Cursor repulsion: distance={cursor_repulsion_distance}, strength={cursor_repulsion_strength}")

# ------------------------- VECTORIZED WAVEFUNCTION FUNCTIONS -------------------------
def R_nl_vec(n, l, r):
    if n <= l:
        return np.zeros_like(r)
    rho = 2.0 * r / n
    L = genlaguerre(n - l - 1, 2*l + 1)(rho)
    const = math.sqrt((2.0/n)**3 * math.factorial(n-l-1) / (2.0*n * math.factorial(n+l)))
    return const * (rho**l) * np.exp(-rho/2.0) * L

def Y_lm_vec(l, m, theta):
    if l == 0:
        return np.full_like(theta, 1.0/math.sqrt(4.0*math.pi))
    elif l == 1:
        if m == 0:
            return math.sqrt(3.0/(4.0*math.pi)) * np.cos(theta)
        elif m == 1:
            return math.sqrt(3.0/(4.0*math.pi)) * np.sin(theta)
    elif l == 2:
        if m == 0:
            return math.sqrt(5.0/(4.0*math.pi)) * (3.0*np.cos(theta)**2 - 1.0)
        elif m == 1:
            return math.sqrt(15.0/(4.0*math.pi)) * np.sin(theta) * np.cos(theta)
        elif m == 2:
            return math.sqrt(15.0/(4.0*math.pi)) * (np.sin(theta)**2)
    elif l == 3:
        if m == 0:
            return math.sqrt(7.0/(4.0*math.pi)) * (5.0*np.cos(theta)**3 - 3.0*np.cos(theta))
        elif m == 1:
            return math.sqrt(21.0/(8.0*math.pi)) * np.sin(theta) * (5.0*np.cos(theta)**2 - 1.0)
        elif m == 2:
            return math.sqrt(105.0/(4.0*math.pi)) * (np.sin(theta)**2 * np.cos(theta))
        elif m == 3:
            return math.sqrt(35.0/(8.0*math.pi)) * (np.sin(theta)**3)
    return np.zeros_like(theta)

def wavefunction_vec(n, l, m, r, alpha):
    theta = np.arccos(np.sin(alpha))  # in the x-z plane (y=0)
    return R_nl_vec(n, l, r) * Y_lm_vec(l, m, theta)

def probability_phase_vec(n, l, m, r, alpha):
    psi = wavefunction_vec(n, l, m, r, alpha)
    prob = psi**2
    phase = np.sign(psi).astype(np.int32)
    return prob, phase

# ------------------------- VECTORIZED FIND_FMAX -------------------------
def find_fmax_vec(n, l, m, rmax, steps_r=300, steps_a=300):
    r_vals = np.linspace(0, rmax, steps_r)
    alpha_vals = np.linspace(0, 2*math.pi, steps_a)
    R, A = np.meshgrid(r_vals, alpha_vals, indexing='ij')
    prob, _ = probability_phase_vec(n, l, m, R, A)
    vals = prob * R
    return np.max(vals)

# ------------------------- VECTORIZED SAMPLING -------------------------
def sample_points_vec(n, l, m, num_points, rmax, fmax):
    two_pi = 2*math.pi
    pts = []
    batch_size = num_points * 10
    while len(pts) < num_points:
        r_cand = rmax * np.random.random(batch_size)
        alpha_cand = two_pi * np.random.random(batch_size)
        prob_cand, phase_cand = probability_phase_vec(n, l, m, r_cand, alpha_cand)
        accept = np.random.random(batch_size) * fmax <= prob_cand * r_cand
        indices = np.where(accept)[0]
        for i in indices:
            ax = r_cand[i] * math.cos(alpha_cand[i])
            az = r_cand[i] * math.sin(alpha_cand[i])
            vx = random.uniform(-movement_strength, movement_strength)
            vz = random.uniform(-movement_strength, movement_strength)
            pts.append((ax, az, ax, az, vx, vz, prob_cand[i], phase_cand[i], ax, az))
            if len(pts) >= num_points:
                break
    return np.array(pts, dtype=np.float64)

# Global particle data array (shape: (num_points, 10))
# Columns: 0-anchor_x, 1-anchor_z, 2-x, 3-z, 4-vx, 5-vz, 6-prob, 7-phase, 8-target_anchor_x, 9-target_anchor_z
particle_data = sample_points_vec(n, l, m, num_points, rmax, find_fmax_vec(n, l, m, rmax))
max_prob = particle_data[:, 6].max() if particle_data.size > 0 else 1e-6

# ------------------------- COLOR SCHEME FUNCTIONS -------------------------
def get_color_original(prob, phase):
    frac = prob / max_prob if max_prob > 1e-14 else 0.0
    brightness = min(255, int(255 * math.sqrt(frac) * brightness_scale))
    if phase > 0:
        return (brightness, brightness, 0)
    elif phase < 0:
        return (brightness, 0, brightness)
    else:
        return (0, 0, 0)

color_stops_gradient = [
    (0.0, (255, 0, 255)),    # bright purple
    (0.5, (255, 165, 0)),     # bright orange
    (1.0, (255, 255, 255))    # white
]

color_stops_hot = [
    (0.0, (255, 0, 0)),      # red
    (0.5, (255, 255, 0)),    # yellow
    (1.0, (255, 255, 255))   # white
]

def interpolate_color(c1, c2, ratio):
    return (int(c1[0] + ratio * (c2[0] - c1[0])),
            int(c1[1] + ratio * (c2[1] - c1[1])),
            int(c1[2] + ratio * (c2[2] - c1[2])))

def get_color_gradient(prob):
    global max_prob
    if max_prob < 1e-14:
        return (255, 255, 255)
    frac = prob / max_prob
    frac = max(0.0, min(1.0, frac))
    for i in range(len(color_stops_gradient) - 1):
        f1, c1 = color_stops_gradient[i]
        f2, c2 = color_stops_gradient[i + 1]
        if frac >= f1 and frac <= f2:
            sub_ratio = (frac - f1) / (f2 - f1)
            return interpolate_color(c1, c2, sub_ratio)
    return color_stops_gradient[-1][1]

def get_color_hot(prob):
    global max_prob
    if max_prob < 1e-14:
        return (255, 255, 255)
    frac = prob / max_prob
    frac = max(0.0, min(1.0, frac))
    for i in range(len(color_stops_hot) - 1):
        f1, c1 = color_stops_hot[i]
        f2, c2 = color_stops_hot[i + 1]
        if frac >= f1 and frac <= f2:
            sub_ratio = (frac - f1) / (f2 - f1)
            return interpolate_color(c1, c2, sub_ratio)
    return color_stops_hot[-1][1]

current_color_scheme = "original"

def choose_color(prob, phase):
    if current_color_scheme == "original":
        return get_color_original(prob, phase)
    elif current_color_scheme == "gradient":
        return get_color_gradient(prob)
    elif current_color_scheme == "hot":
        return get_color_hot(prob)
    else:
        return get_color_original(prob, phase)

# ------------------------- TRANSITION FUNCTION -------------------------
def reinitialize_particles(new_n, new_l, new_m):
    global n, l, m, particle_data, max_prob
    n, l, m = new_n, new_l, new_m
    f_max_new = find_fmax_vec(n, l, m, rmax)
    new_points = sample_points_vec(n, l, m, particle_data.shape[0], rmax, f_max_new)
    particle_data[:, 8] = new_points[:, 0]  # target_anchor_x
    particle_data[:, 9] = new_points[:, 1]  # target_anchor_z
    particle_data[:, 6] = new_points[:, 6]  # probability
    particle_data[:, 7] = new_points[:, 7]  # phase
    max_prob = particle_data[:, 6].max()
    print(f"Transitioned to new parameters: n={n}, l={l}, m={m}")

# ------------------------- PYGAME INTERFACE SETUP -------------------------
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hydrogen Orbital Simulation with Interface")
font = pygame.font.Font(None, 24)
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')

input_boxes = {
    "n": {"rect": pygame.Rect(50, 20, 110, 30), "color": COLOR_INACTIVE, "text": str(n), "active": False, "label": "n:"},
    "l": {"rect": pygame.Rect(50, 60, 110, 30), "color": COLOR_INACTIVE, "text": str(l), "active": False, "label": "l:"},
    "m": {"rect": pygame.Rect(50, 100, 110, 30), "color": COLOR_INACTIVE, "text": str(m), "active": False, "label": "m:"}
}
generate_button = {"rect": pygame.Rect(50, 140, 110, 30), "color": pygame.Color('lightgreen'), "text": "Generate"}

color_scheme_buttons = {
    "original": {"rect": pygame.Rect(50, 180, 30, 30), "color": (255,255,0)},
    "gradient": {"rect": pygame.Rect(90, 180, 30, 30), "color": (255,165,0)},
    "hot": {"rect": pygame.Rect(130, 180, 30, 30), "color": (255,0,0)}
}

zoom = 1.5
base_scale = (WIDTH / 2) / rmax
center_x, center_y = WIDTH//2, HEIGHT//2

def to_screen(xwave, zwave):
    s = base_scale * zoom
    sx = xwave * s + center_x
    sy = -zwave * s + center_y
    return int(sx), int(sy)

def screen_to_wave(sx, sy):
    s = base_scale * zoom
    xwave = (sx - center_x) / s
    zwave = -(sy - center_y) / s
    return xwave, zwave

# ------------------------- VECTORIZED UPDATE FUNCTION -------------------------
def update_particles(data, mouse_x, mouse_z):
    # Update anchor positions toward target anchors
    dx_anchor = data[:, 8] - data[:, 0]
    dz_anchor = data[:, 9] - data[:, 1]
    dist_anchor = np.sqrt(dx_anchor**2 + dz_anchor**2)
    mask_anchor = dist_anchor > transition_speed
    factor = np.empty_like(dist_anchor)
    factor[mask_anchor] = transition_speed / dist_anchor[mask_anchor]
    factor[~mask_anchor] = 1.0
    data[:, 0] += dx_anchor * factor
    data[:, 1] += dz_anchor * factor

    N = data.shape[0]
    # Apply random acceleration
    data[:, 4] += np.random.uniform(-random_accel_strength, random_accel_strength, size=N)
    data[:, 5] += np.random.uniform(-random_accel_strength, random_accel_strength, size=N)
    
    # Apply repulsion from the mouse cursor
    dx_mouse = data[:, 2] - mouse_x
    dz_mouse = data[:, 3] - mouse_z
    dist_mouse = np.sqrt(dx_mouse**2 + dz_mouse**2)
    repulse_mask = dist_mouse < cursor_repulsion_distance
    safe_dist = np.where(dist_mouse < 1e-6, 1e-6, dist_mouse)
    repulsion_acc = cursor_repulsion_strength * (cursor_repulsion_distance - dist_mouse) / safe_dist
    repulsion_acc = repulsion_acc * repulse_mask  # zero force outside repulsion distance
    data[:, 4] += repulsion_acc * dx_mouse
    data[:, 5] += repulsion_acc * dz_mouse

    # Cap maximum speed only for particles NOT being repelled by the cursor
    speed = np.sqrt(data[:, 4]**2 + data[:, 5]**2)
    mask_speed = (speed > max_speed) & (dist_mouse >= cursor_repulsion_distance)
    if np.any(mask_speed):
        data[mask_speed, 4] *= (max_speed / speed[mask_speed])
        data[mask_speed, 5] *= (max_speed / speed[mask_speed])
    
    # Update positions based on velocity
    data[:, 2] += data[:, 4]
    data[:, 3] += data[:, 5]
    
    # Constrain particles so they don’t drift too far from their anchor
    dxp = data[:, 2] - data[:, 0]
    dzp = data[:, 3] - data[:, 1]
    drift = np.sqrt(dxp**2 + dzp**2)
    mask_drift = drift > max_drift
    if np.any(mask_drift):
        data[mask_drift, 2] = data[mask_drift, 0] + dxp[mask_drift]*(max_drift/drift[mask_drift])
        data[mask_drift, 3] = data[mask_drift, 1] + dzp[mask_drift]*(max_drift/drift[mask_drift])
        nx = np.zeros_like(dxp)
        nz = np.zeros_like(dzp)
        nx[mask_drift] = dxp[mask_drift] / drift[mask_drift]
        nz[mask_drift] = dzp[mask_drift] / drift[mask_drift]
        dot = data[mask_drift, 4] * nx[mask_drift] + data[mask_drift, 5] * nz[mask_drift]
        data[mask_drift, 4] -= 2 * dot * nx[mask_drift]
        data[mask_drift, 5] -= 2 * dot * nz[mask_drift]

# ------------------------- INTERFACE EVENT HANDLING -------------------------
def handle_interface_events(event):
    global n, l, m, current_color_scheme
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = event.pos
        for box in input_boxes.values():
            if box["rect"].collidepoint(pos):
                box["active"] = True
                box["color"] = COLOR_ACTIVE
            else:
                box["active"] = False
                box["color"] = COLOR_INACTIVE
        if generate_button["rect"].collidepoint(pos):
            try:
                new_n = int(input_boxes["n"]["text"])
                new_l = int(input_boxes["l"]["text"])
                new_m = int(input_boxes["m"]["text"])
                if new_n > new_l and abs(new_m) <= new_l:
                    reinitialize_particles(new_n, new_l, new_m)
                else:
                    print("Invalid quantum numbers (require n>l and |m|<=l).")
            except ValueError:
                print("Enter valid integers for n, l, m.")
        for scheme, button in color_scheme_buttons.items():
            if button["rect"].collidepoint(pos):
                current_color_scheme = scheme
                print(f"Color scheme set to {scheme}")
    elif event.type == pygame.KEYDOWN:
        for box in input_boxes.values():
            if box["active"]:
                if event.key == pygame.K_RETURN:
                    box["active"] = False
                    box["color"] = COLOR_INACTIVE
                elif event.key == pygame.K_BACKSPACE:
                    box["text"] = box["text"][:-1]
                else:
                    if event.unicode.isdigit() or (box["label"] == "m:" and event.unicode == "-"):
                        box["text"] += event.unicode

current_color_scheme = "original"

def choose_color(prob, phase):
    if current_color_scheme == "original":
        return get_color_original(prob, phase)
    elif current_color_scheme == "gradient":
        return get_color_gradient(prob)
    elif current_color_scheme == "hot":
        return get_color_hot(prob)
    else:
        return get_color_original(prob, phase)

# ------------------------- MAIN LOOP -------------------------
clock = pygame.time.Clock()
running = True
while running:
    dt = clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            handle_interface_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                zoom *= 1.1
            elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                zoom = max(0.05, zoom/1.1)
    
    # Get current mouse position (screen coordinates) and convert to simulation coordinates
    mouse_screen = pygame.mouse.get_pos()
    mouse_wave_x, mouse_wave_z = screen_to_wave(mouse_screen[0], mouse_screen[1])
    
    update_particles(particle_data, mouse_wave_x, mouse_wave_z)
    
    screen.fill((0, 0, 0))
    for i in range(particle_data.shape[0]):
        prob = particle_data[i, 6]
        col = choose_color(prob, particle_data[i, 7])
        sx, sy = to_screen(particle_data[i, 2], particle_data[i, 3])
        pygame.draw.circle(screen, col, (sx, sy), 1)
    for key, box in input_boxes.items():
        pygame.draw.rect(screen, box["color"], box["rect"], 2)
        label_surf = font.render(box["label"], True, pygame.Color('white'))
        screen.blit(label_surf, (box["rect"].x - 30, box["rect"].y + 5))
        text_surf = font.render(box["text"], True, pygame.Color('white'))
        screen.blit(text_surf, (box["rect"].x + 5, box["rect"].y + 5))
    pygame.draw.rect(screen, generate_button["color"], generate_button["rect"])
    btn_text = font.render(generate_button["text"], True, pygame.Color('black'))
    btn_rect = btn_text.get_rect(center=generate_button["rect"].center)
    screen.blit(btn_text, btn_rect)
    for scheme, button in color_scheme_buttons.items():
        pygame.draw.rect(screen, button["color"], button["rect"])
    pygame.display.flip()
pygame.quit()
sys.exit()
