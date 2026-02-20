import pygame
import sys
import math
import random
import numpy as np
from scipy.special import genlaguerre
n = 3
l = 2
m = 0
brightness_scale = 50.0
num_points = 10000
rmax = 50.0
max_drift = 2.0
movement_strength = 0.1
random_accel_strength = 0.02
max_speed = 0.05
transition_speed = 0.5
print(f'Initial: Hydrogen Orbital (n={n}, l={l}, m={m})')
print(f'Movement strength={movement_strength}, max_drift={max_drift}, random accel={random_accel_strength}, max_speed={max_speed}, transition_speed={transition_speed}')

def R_nl_vec(n, l, r):
    if n <= l:
        return np.zeros_like(r)
    rho = 2.0 * r / n
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    const = math.sqrt((2.0 / n) ** 3 * math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l)))
    return const * rho ** l * np.exp(-rho / 2.0) * L

def Y_lm_vec(l, m, theta, phi=None):
    if phi is None:
        if l == 0:
            return np.full_like(theta, 1.0 / math.sqrt(4.0 * math.pi))
        elif l == 1:
            if m == 0:
                return math.sqrt(3.0 / (4.0 * math.pi)) * np.cos(theta)
            elif m == 1:
                return math.sqrt(3.0 / (4.0 * math.pi)) * np.sin(theta)
        elif l == 2:
            if m == 0:
                return math.sqrt(5.0 / (4.0 * math.pi)) * (3.0 * np.cos(theta) ** 2 - 1.0)
            elif m == 1:
                return math.sqrt(15.0 / (4.0 * math.pi)) * np.sin(theta) * np.cos(theta)
            elif m == 2:
                return math.sqrt(15.0 / (4.0 * math.pi)) * np.sin(theta) ** 2
        elif l == 3:
            if m == 0:
                return math.sqrt(7.0 / (4.0 * math.pi)) * (5.0 * np.cos(theta) ** 3 - 3.0 * np.cos(theta))
            elif m == 1:
                return math.sqrt(21.0 / (8.0 * math.pi)) * np.sin(theta) * (5.0 * np.cos(theta) ** 2 - 1.0)
            elif m == 2:
                return math.sqrt(105.0 / (4.0 * math.pi)) * (np.sin(theta) ** 2 * np.cos(theta))
            elif m == 3:
                return math.sqrt(35.0 / (8.0 * math.pi)) * np.sin(theta) ** 3
        return np.zeros_like(theta)
    else:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        if l == 0:
            return np.full_like(theta, 1.0 / math.sqrt(4.0 * math.pi))
        elif l == 1:
            if m == 0:
                return math.sqrt(3.0 / (4.0 * math.pi)) * cos_theta
            elif abs(m) == 1:
                if m > 0:
                    return math.sqrt(3.0 / (8.0 * math.pi)) * sin_theta * np.cos(phi)
                else:
                    return math.sqrt(3.0 / (8.0 * math.pi)) * sin_theta * np.sin(phi)
        elif l == 2:
            if m == 0:
                return math.sqrt(5.0 / (4.0 * math.pi)) * (3.0 * cos_theta ** 2 - 1.0) / 2.0
            elif abs(m) == 1:
                if m > 0:
                    return math.sqrt(15.0 / (8.0 * math.pi)) * sin_theta * cos_theta * np.cos(phi)
                else:
                    return math.sqrt(15.0 / (8.0 * math.pi)) * sin_theta * cos_theta * np.sin(phi)
            elif abs(m) == 2:
                if m > 0:
                    return math.sqrt(15.0 / (32.0 * math.pi)) * sin_theta ** 2 * np.cos(2 * phi)
                else:
                    return math.sqrt(15.0 / (32.0 * math.pi)) * sin_theta ** 2 * np.sin(2 * phi)
        elif l == 3:
            if m == 0:
                return math.sqrt(7.0 / (4.0 * math.pi)) * (5.0 * cos_theta ** 3 - 3.0 * cos_theta) / 2.0
            elif abs(m) == 1:
                if m > 0:
                    return math.sqrt(21.0 / (64.0 * math.pi)) * sin_theta * (5.0 * cos_theta ** 2 - 1.0) * np.cos(phi)
                else:
                    return math.sqrt(21.0 / (64.0 * math.pi)) * sin_theta * (5.0 * cos_theta ** 2 - 1.0) * np.sin(phi)
            elif abs(m) == 2:
                if m > 0:
                    return math.sqrt(105.0 / (32.0 * math.pi)) * sin_theta ** 2 * cos_theta * np.cos(2 * phi)
                else:
                    return math.sqrt(105.0 / (32.0 * math.pi)) * sin_theta ** 2 * cos_theta * np.sin(2 * phi)
            elif abs(m) == 3:
                if m > 0:
                    return math.sqrt(35.0 / (64.0 * math.pi)) * sin_theta ** 3 * np.cos(3 * phi)
                else:
                    return math.sqrt(35.0 / (64.0 * math.pi)) * sin_theta ** 3 * np.sin(3 * phi)
        return np.zeros_like(theta)

def wavefunction_vec(n, l, m, r, alpha, phi=None):
    if phi is None:
        theta = np.arccos(np.sin(alpha))
        return R_nl_vec(n, l, r) * Y_lm_vec(l, m, theta, None)
    else:
        return R_nl_vec(n, l, r) * Y_lm_vec(l, m, alpha, phi)

def probability_phase_vec(n, l, m, r, alpha, phi=None):
    psi = wavefunction_vec(n, l, m, r, alpha, phi)
    prob = psi ** 2
    phase = np.sign(psi).astype(np.int32)
    return (prob, phase)

def find_fmax_vec(n, l, m, rmax, steps_r=300, steps_a=300, use_3d=False):
    r_vals = np.linspace(0, rmax, steps_r)
    if use_3d:
        samples = max(50000, steps_r * steps_a * 2)
        r_cand = rmax * np.random.random(samples)
        theta_cand = np.arccos(1.0 - 2.0 * np.random.random(samples))
        phi_cand = 2 * math.pi * np.random.random(samples)
        prob, _ = probability_phase_vec(n, l, m, r_cand, theta_cand, phi_cand)
        vals = prob * r_cand ** 2 * np.sin(theta_cand)
    else:
        alpha_vals = np.linspace(0, 2 * math.pi, steps_a)
        R, A = np.meshgrid(r_vals, alpha_vals, indexing='ij')
        prob, _ = probability_phase_vec(n, l, m, R, A, None)
        vals = prob * R
    return float(np.max(vals)) * 1.25

def sample_points_vec(n, l, m, num_points, rmax, fmax, use_3d=False):
    two_pi = 2 * math.pi
    pts = []
    batch_size = num_points * 10
    while len(pts) < num_points:
        r_cand = rmax * np.random.random(batch_size)
        if use_3d:
            theta_cand = np.arccos(1.0 - 2.0 * np.random.random(batch_size))
            phi_cand = two_pi * np.random.random(batch_size)
            prob_cand, phase_cand = probability_phase_vec(n, l, m, r_cand, theta_cand, phi_cand)
            accept = np.random.random(batch_size) * fmax <= prob_cand * r_cand ** 2 * np.sin(theta_cand)
            indices = np.where(accept)[0]
            for i in indices:
                ax = r_cand[i] * np.sin(theta_cand[i]) * np.cos(phi_cand[i])
                ay = r_cand[i] * np.sin(theta_cand[i]) * np.sin(phi_cand[i])
                az = r_cand[i] * np.cos(theta_cand[i])
                vx = random.uniform(-movement_strength, movement_strength)
                vy = random.uniform(-movement_strength, movement_strength)
                vz = random.uniform(-movement_strength, movement_strength)
                pts.append((ax, ay, az, ax, ay, az, vx, vy, vz, prob_cand[i], phase_cand[i], ax, ay, az))
                if len(pts) >= num_points:
                    break
        else:
            alpha_cand = two_pi * np.random.random(batch_size)
            prob_cand, phase_cand = probability_phase_vec(n, l, m, r_cand, alpha_cand, None)
            accept = np.random.random(batch_size) * fmax <= prob_cand * r_cand
            indices = np.where(accept)[0]
            for i in indices:
                ax = r_cand[i] * math.cos(alpha_cand[i])
                az = r_cand[i] * math.sin(alpha_cand[i])
                vx = random.uniform(-movement_strength, movement_strength)
                vz = random.uniform(-movement_strength, movement_strength)
                pts.append((ax, 0.0, az, ax, 0.0, az, vx, 0.0, vz, prob_cand[i], phase_cand[i], ax, 0.0, az))
                if len(pts) >= num_points:
                    break
    return np.array(pts, dtype=np.float64)
view_3d = False
particle_data = sample_points_vec(n, l, m, num_points, rmax, find_fmax_vec(n, l, m, rmax, use_3d=False), use_3d=False)
max_prob = particle_data[:, 9].max() if particle_data.size > 0 else 1e-06
color_stops_gradient = [(0.0, (255, 0, 255)), (0.5, (255, 165, 0)), (1.0, (255, 255, 255))]

def interpolate_color(c1, c2, ratio):
    return (int(c1[0] + ratio * (c2[0] - c1[0])), int(c1[1] + ratio * (c2[1] - c1[1])), int(c1[2] + ratio * (c2[2] - c1[2])))

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

def choose_color(prob, phase=None):
    return get_color_gradient(prob)

def reinitialize_particles(new_n, new_l, new_m):
    global n, l, m, particle_data, max_prob, view_3d
    n, l, m = (new_n, new_l, new_m)
    f_max_new = find_fmax_vec(n, l, m, rmax, use_3d=view_3d)
    new_points = sample_points_vec(n, l, m, particle_data.shape[0], rmax, f_max_new, use_3d=view_3d)
    particle_data[:, 11] = new_points[:, 0]
    particle_data[:, 12] = new_points[:, 1]
    particle_data[:, 13] = new_points[:, 2]
    particle_data[:, 9] = new_points[:, 9]
    particle_data[:, 10] = new_points[:, 10]
    max_prob = particle_data[:, 9].max()
    print(f'Transitioned to new parameters: n={n}, l={l}, m={m}')
pygame.init()
WIDTH, HEIGHT = (1280, 720)
is_fullscreen = False
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hydrogen Orbital Simulation with Interface')
font = pygame.font.Font(None, 24)
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')
input_boxes = {'n': {'rect': pygame.Rect(50, 20, 110, 30), 'color': COLOR_INACTIVE, 'text': str(n), 'active': False, 'label': 'n:'}, 'l': {'rect': pygame.Rect(50, 60, 110, 30), 'color': COLOR_INACTIVE, 'text': str(l), 'active': False, 'label': 'l:'}, 'm': {'rect': pygame.Rect(50, 100, 110, 30), 'color': COLOR_INACTIVE, 'text': str(m), 'active': False, 'label': 'm:'}}
generate_button = {'rect': pygame.Rect(50, 140, 110, 30), 'color': pygame.Color('lightgreen'), 'text': 'Generate'}
view_3d_button = {'rect': pygame.Rect(50, 180, 110, 30), 'color': pygame.Color('lightblue'), 'text': '3D View'}
rotation_x = 0.0
rotation_y = 0.0
is_rotating = False
last_mouse_pos = (0, 0)
rotation_sensitivity = 0.006
zoom = 1.5
base_scale = WIDTH / 2 / rmax
center_x, center_y = (WIDTH // 2, HEIGHT // 2)

def rebuild_display(fullscreen: bool):
    global screen, WIDTH, HEIGHT, base_scale, center_x, center_y, is_fullscreen
    is_fullscreen = fullscreen
    if fullscreen:
        info = pygame.display.Info()
        WIDTH, HEIGHT = (info.current_w, info.current_h)
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    else:
        WIDTH, HEIGHT = (1280, 720)
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    base_scale = WIDTH / 2 / rmax
    center_x, center_y = (WIDTH // 2, HEIGHT // 2)

def auto_fit_zoom_2d(margin_px: int=40):
    global zoom
    if particle_data is None or particle_data.size == 0:
        zoom = 1.0
        return
    xs = particle_data[:, 3]
    zs = particle_data[:, 5]
    max_extent = float(max(np.max(np.abs(xs)), np.max(np.abs(zs)), 1e-06))
    available = max(50.0, min(WIDTH, HEIGHT) - 2 * margin_px)
    target_half = available * 0.5
    zoom = target_half / (base_scale * max_extent)
    zoom = max(0.05, min(10.0, zoom))
if not view_3d:
    auto_fit_zoom_2d()

def to_screen(xwave, zwave):
    s = base_scale * zoom
    sx = xwave * s + center_x
    sy = -zwave * s + center_y
    return (int(sx), int(sy))

def screen_to_wave(sx, sy):
    s = base_scale * zoom
    xwave = (sx - center_x) / s
    zwave = -(sy - center_y) / s
    return (xwave, zwave)

def rotate_3d(x, y, z, rot_x, rot_y):
    cos_x, sin_x = (math.cos(rot_x), math.sin(rot_x))
    y_new = y * cos_x - z * sin_x
    z_new = y * sin_x + z * cos_x
    y, z = (y_new, z_new)
    cos_y, sin_y = (math.cos(rot_y), math.sin(rot_y))
    x_new = x * cos_y + z * sin_y
    z_new = -x * sin_y + z * cos_y
    x, z = (x_new, z_new)
    return (x, y, z)

def project_3d_to_screen(x, y, z, rot_x, rot_y):
    x_rot, y_rot, z_rot = rotate_3d(x, y, z, rot_x, rot_y)
    s = base_scale * zoom
    sx = x_rot * s + center_x
    sy = -z_rot * s + center_y
    return (int(sx), int(sy), y_rot)

def draw_grid_3d(surface, rot_x, rot_y, extent=50.0, spacing=5.0):
    grid_col = (35, 35, 35)
    steps = int(2 * extent / spacing)
    for i in range(steps + 1):
        x = -extent + i * spacing
        sx1, sy1, _ = project_3d_to_screen(x, 0.0, -extent, rot_x, rot_y)
        sx2, sy2, _ = project_3d_to_screen(x, 0.0, extent, rot_x, rot_y)
        pygame.draw.line(surface, grid_col, (sx1, sy1), (sx2, sy2), 1)
        z = -extent + i * spacing
        sx1, sy1, _ = project_3d_to_screen(-extent, 0.0, z, rot_x, rot_y)
        sx2, sy2, _ = project_3d_to_screen(extent, 0.0, z, rot_x, rot_y)
        pygame.draw.line(surface, grid_col, (sx1, sy1), (sx2, sy2), 1)

def update_particles(data, mouse_x, mouse_z, mouse_y=0.0, use_3d=False):
    dx_anchor = data[:, 11] - data[:, 0]
    dy_anchor = data[:, 12] - data[:, 1]
    dz_anchor = data[:, 13] - data[:, 2]
    dist_anchor = np.sqrt(dx_anchor ** 2 + dy_anchor ** 2 + dz_anchor ** 2)
    mask_anchor = dist_anchor > transition_speed
    factor = np.empty_like(dist_anchor)
    factor[mask_anchor] = transition_speed / dist_anchor[mask_anchor]
    factor[~mask_anchor] = 1.0
    data[:, 0] += dx_anchor * factor
    data[:, 1] += dy_anchor * factor
    data[:, 2] += dz_anchor * factor
    N = data.shape[0]
    data[:, 6] += np.random.uniform(-random_accel_strength, random_accel_strength, size=N)
    data[:, 7] += np.random.uniform(-random_accel_strength, random_accel_strength, size=N)
    data[:, 8] += np.random.uniform(-random_accel_strength, random_accel_strength, size=N)
    if use_3d:
        speed = np.sqrt(data[:, 6] ** 2 + data[:, 7] ** 2 + data[:, 8] ** 2)
        mask_speed = speed > max_speed
        if np.any(mask_speed):
            data[mask_speed, 6] *= max_speed / speed[mask_speed]
            data[mask_speed, 7] *= max_speed / speed[mask_speed]
            data[mask_speed, 8] *= max_speed / speed[mask_speed]
        data[:, 3] += data[:, 6]
        data[:, 4] += data[:, 7]
        data[:, 5] += data[:, 8]
        dxp = data[:, 3] - data[:, 0]
        dyp = data[:, 4] - data[:, 1]
        dzp = data[:, 5] - data[:, 2]
        drift = np.sqrt(dxp ** 2 + dyp ** 2 + dzp ** 2)
        mask_drift = drift > max_drift
        if np.any(mask_drift):
            data[mask_drift, 3] = data[mask_drift, 0] + dxp[mask_drift] * (max_drift / drift[mask_drift])
            data[mask_drift, 4] = data[mask_drift, 1] + dyp[mask_drift] * (max_drift / drift[mask_drift])
            data[mask_drift, 5] = data[mask_drift, 2] + dzp[mask_drift] * (max_drift / drift[mask_drift])
            nx = np.zeros_like(dxp)
            ny = np.zeros_like(dyp)
            nz = np.zeros_like(dzp)
            nx[mask_drift] = dxp[mask_drift] / drift[mask_drift]
            ny[mask_drift] = dyp[mask_drift] / drift[mask_drift]
            nz[mask_drift] = dzp[mask_drift] / drift[mask_drift]
            dot = data[mask_drift, 6] * nx[mask_drift] + data[mask_drift, 7] * ny[mask_drift] + data[mask_drift, 8] * nz[mask_drift]
            data[mask_drift, 6] -= 2 * dot * nx[mask_drift]
            data[mask_drift, 7] -= 2 * dot * ny[mask_drift]
            data[mask_drift, 8] -= 2 * dot * nz[mask_drift]
    else:
        speed = np.sqrt(data[:, 6] ** 2 + data[:, 8] ** 2)
        mask_speed = speed > max_speed
        if np.any(mask_speed):
            data[mask_speed, 6] *= max_speed / speed[mask_speed]
            data[mask_speed, 8] *= max_speed / speed[mask_speed]
        data[:, 3] += data[:, 6]
        data[:, 5] += data[:, 8]
        dxp = data[:, 3] - data[:, 0]
        dzp = data[:, 5] - data[:, 2]
        drift = np.sqrt(dxp ** 2 + dzp ** 2)
        mask_drift = drift > max_drift
        if np.any(mask_drift):
            data[mask_drift, 3] = data[mask_drift, 0] + dxp[mask_drift] * (max_drift / drift[mask_drift])
            data[mask_drift, 5] = data[mask_drift, 2] + dzp[mask_drift] * (max_drift / drift[mask_drift])
            nx = np.zeros_like(dxp)
            nz = np.zeros_like(dzp)
            nx[mask_drift] = dxp[mask_drift] / drift[mask_drift]
            nz[mask_drift] = dzp[mask_drift] / drift[mask_drift]
            dot = data[mask_drift, 6] * nx[mask_drift] + data[mask_drift, 8] * nz[mask_drift]
            data[mask_drift, 6] -= 2 * dot * nx[mask_drift]
            data[mask_drift, 8] -= 2 * dot * nz[mask_drift]

def handle_interface_events(event):
    global n, l, m, view_3d, is_rotating, last_mouse_pos, rotation_x, rotation_y, particle_data, max_prob, zoom
    ui_rects = [box['rect'] for box in input_boxes.values()] + [generate_button['rect'], view_3d_button['rect']]
    if event.type == pygame.MOUSEWHEEL:
        zoom *= 1.1 ** event.y
        zoom = max(0.05, min(10.0, zoom))
        return
    if event.type == pygame.MOUSEBUTTONDOWN:
        if getattr(event, 'button', None) == 4:
            zoom *= 1.1
            zoom = max(0.05, min(10.0, zoom))
            return
        if getattr(event, 'button', None) == 5:
            zoom = max(0.05, zoom / 1.1)
            return
        pos = event.pos
        clicked_ui = any((r.collidepoint(pos) for r in ui_rects))
        if view_3d and getattr(event, 'button', None) in (1, 3) and (not clicked_ui):
            is_rotating = True
            last_mouse_pos = pos
            return
        for box in input_boxes.values():
            if box['rect'].collidepoint(pos):
                box['active'] = True
                box['color'] = COLOR_ACTIVE
            else:
                box['active'] = False
                box['color'] = COLOR_INACTIVE
        if generate_button['rect'].collidepoint(pos):
            try:
                new_n = int(input_boxes['n']['text'])
                new_l = int(input_boxes['l']['text'])
                new_m = int(input_boxes['m']['text'])
                if new_n > new_l and abs(new_m) <= new_l:
                    reinitialize_particles(new_n, new_l, new_m)
                    if not view_3d:
                        auto_fit_zoom_2d()
                else:
                    print('Invalid quantum numbers (require n>l and |m|<=l).')
            except ValueError:
                print('Enter valid integers for n, l, m.')
        if view_3d_button['rect'].collidepoint(pos):
            view_3d = not view_3d
            view_3d_button['text'] = '2D View' if view_3d else '3D View'
            print(f'Switched to {('3D' if view_3d else '2D')} mode')
            f_max_new = find_fmax_vec(n, l, m, rmax, use_3d=view_3d)
            new_points = sample_points_vec(n, l, m, particle_data.shape[0], rmax, f_max_new, use_3d=view_3d)
            particle_data = new_points
            max_prob = particle_data[:, 9].max() if particle_data.size > 0 else 1e-06
            if not view_3d:
                auto_fit_zoom_2d()
    elif event.type == pygame.MOUSEBUTTONUP:
        if getattr(event, 'button', None) in (1, 3):
            is_rotating = False
    elif event.type == pygame.MOUSEMOTION:
        if is_rotating and view_3d:
            dx = event.pos[0] - last_mouse_pos[0]
            dy = event.pos[1] - last_mouse_pos[1]
            rotation_y += dx * rotation_sensitivity
            rotation_x += dy * rotation_sensitivity
            rotation_x = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, rotation_x))
            last_mouse_pos = event.pos
    elif event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_F11, pygame.K_f):
            rebuild_display(not is_fullscreen)
            if not view_3d:
                auto_fit_zoom_2d()
        if event.key == pygame.K_r:
            rotation_x = 0.0
            rotation_y = 0.0
        for box in input_boxes.values():
            if box['active']:
                if event.key == pygame.K_RETURN:
                    box['active'] = False
                    box['color'] = COLOR_INACTIVE
                elif event.key == pygame.K_BACKSPACE:
                    box['text'] = box['text'][:-1]
                elif event.unicode.isdigit() or (box['label'] == 'm:' and event.unicode == '-'):
                    box['text'] += event.unicode
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
                zoom = max(0.05, zoom / 1.1)
    mouse_screen = pygame.mouse.get_pos()
    if view_3d:
        mouse_wave_x, mouse_wave_y, mouse_wave_z = (0.0, 0.0, 0.0)
    else:
        mouse_wave_x, mouse_wave_z = screen_to_wave(mouse_screen[0], mouse_screen[1])
        mouse_wave_y = 0.0
    update_particles(particle_data, mouse_wave_x, mouse_wave_z, mouse_wave_y, use_3d=view_3d)
    screen.fill((0, 0, 0))
    if view_3d:
        draw_grid_3d(screen, rotation_x, rotation_y, extent=rmax, spacing=5.0)
        particles_to_render = []
        for i in range(particle_data.shape[0]):
            prob = particle_data[i, 9]
            phase = particle_data[i, 10]
            x, y, z = (particle_data[i, 3], particle_data[i, 4], particle_data[i, 5])
            sx, sy, depth = project_3d_to_screen(x, y, z, rotation_x, rotation_y)
            col = choose_color(prob, phase)
            particles_to_render.append((sx, sy, depth, col))
        particles_to_render.sort(key=lambda p: p[2], reverse=True)
        for sx, sy, depth, col in particles_to_render:
            if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                pygame.draw.circle(screen, col, (sx, sy), 1)
    else:
        for i in range(particle_data.shape[0]):
            prob = particle_data[i, 9]
            phase = particle_data[i, 10]
            col = choose_color(prob, phase)
            sx, sy = to_screen(particle_data[i, 3], particle_data[i, 5])
            pygame.draw.circle(screen, col, (sx, sy), 1)
    for key, box in input_boxes.items():
        pygame.draw.rect(screen, box['color'], box['rect'], 2)
        label_surf = font.render(box['label'], True, pygame.Color('white'))
        screen.blit(label_surf, (box['rect'].x - 30, box['rect'].y + 5))
        text_surf = font.render(box['text'], True, pygame.Color('white'))
        screen.blit(text_surf, (box['rect'].x + 5, box['rect'].y + 5))
    pygame.draw.rect(screen, generate_button['color'], generate_button['rect'])
    btn_text = font.render(generate_button['text'], True, pygame.Color('black'))
    btn_rect = btn_text.get_rect(center=generate_button['rect'].center)
    screen.blit(btn_text, btn_rect)
    pygame.draw.rect(screen, view_3d_button['color'], view_3d_button['rect'])
    btn_3d_text = font.render(view_3d_button['text'], True, pygame.Color('black'))
    btn_3d_rect = btn_3d_text.get_rect(center=view_3d_button['rect'].center)
    screen.blit(btn_3d_text, btn_3d_rect)
    if view_3d:
        info_text = font.render('Right-drag: rotate | Scroll/+/-: zoom | R: reset', True, pygame.Color('white'))
        screen.blit(info_text, (WIDTH - 430, 20))
    pygame.display.flip()
pygame.quit()
sys.exit()
