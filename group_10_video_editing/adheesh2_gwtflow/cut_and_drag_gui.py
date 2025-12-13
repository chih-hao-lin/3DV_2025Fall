from rp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import Polygon as Polygon
import cv2
git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw
from easydict import EasyDict


def select_polygon(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    path = []

    def onclick(event):
        if event.button == 1:  # Left click
            if event.xdata is not None and event.ydata is not None:
                path.append((event.xdata, event.ydata))
            ax.clear()
            ax.imshow(image)
            ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")
            for i in range(len(path)):
                if i > 0:
                    ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "r-")
                ax.plot(path[i][0], path[i][1], "ro")
            if len(path) > 1:
                ax.plot([path[-1][0], path[0][0]], [path[-1][1], path[0][1]], "r--")
            if len(path) > 2:
                polygon = Polygon(path, closed=True, alpha=0.3, facecolor="r", edgecolor="r")
                ax.add_patch(polygon)
            fig.canvas.draw()
        elif event.button == 3 and path:  # Right click
            path.pop()
            ax.clear()
            ax.imshow(image)
            ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")
            for i in range(len(path)):
                if i > 0:
                    ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "r-")
                ax.plot(path[i][0], path[i][1], "ro")
            if len(path) > 1:
                ax.plot([path[-1][0], path[0][0]], [path[-1][1], path[0][1]], "r--")
            if len(path) > 2:
                polygon = Polygon(path, closed=True, alpha=0.3, facecolor="r", edgecolor="r")
                ax.add_patch(polygon)
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return path


def select_polygon_and_path(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    polygon_path = []
    movement_path = []

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return polygon_path, movement_path


def select_path(image, polygon, num_frames=49):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    path = []

    # Add sliders for final scale, in-plane rotation (roll), and vertical-axis spin (yaw)
    ax_scale = plt.axes([0.25, 0.08, 0.65, 0.03])
    ax_rot = plt.axes([0.25, 0.12, 0.65, 0.03])
    ax_yaw = plt.axes([0.25, 0.16, 0.65, 0.03])
    ax_back = plt.axes([0.25, 0.20, 0.2, 0.05])

    scale_slider = Slider(ax_scale, "Final Scale", 0.1, 5.0, valinit=1)
    rot_slider = Slider(ax_rot, "Final Roll (deg)", -360, 360, valinit=0)
    yaw_slider = Slider(ax_yaw, "Final Spin (deg)", 0, 360, valinit=0)
    backfill_toggle = CheckButtons(ax_back, ["Mirror Back"], [True])

    scales = []
    rotations = []
    yaws = []
    mirror_back = True

    def interpolate_transformations(n_points):
        # Exponential scale path for smoother growth; linear angles
        scales = np.exp(np.linspace(0, np.log(scale_slider.val), n_points))
        rotations = np.linspace(0, rot_slider.val, n_points)
        yaws = np.linspace(0, yaw_slider.val, n_points)
        return scales, rotations, yaws

    spin_preview_cache = {}

    def update_display():
        ax.clear()
        ax.imshow(image)
        ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

        n_points = len(path)
        if n_points < 1:
            fig.canvas.draw_idle()
            return

        # Interpolate scales, roll, and spin over the total number of points
        scales[:], rotations[:], yaws[:] = interpolate_transformations(n_points)

        origin = np.array(path[0])

        for i in range(n_points):
            ax.plot(path[i][0], path[i][1], "bo")
            if i > 0:
                ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "b-")
            # If spin is requested, preview using surface-of-revolution silhouette; else use planar
            yaw_i = yaws[i]
            if abs(yaw_i) > 1e-3:
                # Precompute cropped patch and pivot for preview
                if not spin_preview_cache:
                    poly_np = np.array(polygon, dtype=np.float32)
                    H0, W0 = image.shape[:2]
                    x_min = max(0, int(np.floor(np.min(poly_np[:, 0]))))
                    y_min = max(0, int(np.floor(np.min(poly_np[:, 1]))))
                    x_max = min(W0, int(np.ceil(np.max(poly_np[:, 0]))))
                    y_max = min(H0, int(np.ceil(np.max(poly_np[:, 1]))))
                    patch = image[y_min:y_max, x_min:x_max].copy()
                    ph, pw = patch.shape[:2]
                    patch_mask = np.zeros((ph, pw), dtype=np.uint8)
                    shifted_poly = (poly_np - np.array([x_min, y_min], dtype=np.float32)).astype(np.float32)
                    cv2.fillPoly(patch_mask, [shifted_poly.astype(np.int32)], 255)
                    if patch.ndim == 3 and patch.shape[2] == 3:
                        patch_rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
                    else:
                        c = patch.shape[2] if patch.ndim == 3 else 1
                        arr = patch if c >= 3 else np.repeat(patch, 3, axis=2)
                        arr = arr[:, :, :3].copy()
                        patch_rgba = np.concatenate([arr, np.ones((ph, pw, 1), dtype=arr.dtype) * 255], axis=2)
                    patch_rgba[:, :, 3] = patch_mask
                    pivot_in_patch = origin - np.array([x_min, y_min], dtype=np.float32)
                    # Precompute back-face once for fast preview
                    Mflip = np.array([[-1, 0, 2 * float(pivot_in_patch[0])], [0, 1, 0]], dtype=np.float32)
                    patch_back = cv2.warpAffine(
                        patch_rgba,
                        Mflip,
                        (pw, ph),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    spin_preview_cache.update(dict(patch_rgba=patch_rgba, pivot=pivot_in_patch, patch_back=patch_back))

                patch_rgba = spin_preview_cache['patch_rgba']
                pivot_in_patch = spin_preview_cache['pivot']
                patch_back = spin_preview_cache['patch_back']
                spun = spin_patch(
                    patch_rgba,
                    pivot_x=float(pivot_in_patch[0]),
                    yaw_deg=float(yaw_i),
                    fov_deg=60.0,
                    back_flipped_rgba=patch_back,
                    fill_backface=mirror_back,
                )
                roll_i = rotations[i]
                if abs(roll_i) > 1e-3:
                    spun = rotate_patch_around_pivot(spun, pivot_in_patch, roll_i)
                scale_i = float(scales[i])
                if abs(scale_i - 1.0) > 1e-3:
                    spun, pivot_scaled = scale_patch_about_pivot(spun, pivot_in_patch, scale_i)
                else:
                    pivot_scaled = pivot_in_patch.copy()
                dx = float(path[i][0] - pivot_scaled[0])
                dy = float(path[i][1] - pivot_scaled[1])
                mask = (spun[:, :, 3] > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    transformed_polygon = cnt.reshape(-1, 2).astype(np.float32)
                    transformed_polygon[:, 0] += dx
                    transformed_polygon[:, 1] += dy
                    mpl_poly = Polygon(
                        transformed_polygon,
                        closed=True,
                        alpha=0.3,
                        facecolor="r",
                        edgecolor="r",
                    )
                    ax.add_patch(mpl_poly)
            else:
                # Planar preview
                transformed_polygon = apply_transformation(np.array(polygon), scales[i], rotations[i], origin)
                position_offset = np.array(path[i]) - origin
                transformed_polygon += position_offset
                mpl_poly = Polygon(
                    transformed_polygon,
                    closed=True,
                    alpha=0.3,
                    facecolor="r",
                    edgecolor="r",
                )
                ax.add_patch(mpl_poly)

        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click
            path.append((event.xdata, event.ydata))
            update_display()
        elif event.button == 3 and path:  # Right click
            path.pop()
            update_display()

    def on_slider_change(val):
        update_display()

    scale_slider.on_changed(on_slider_change)
    rot_slider.on_changed(on_slider_change)
    yaw_slider.on_changed(on_slider_change)
    def on_backfill_clicked(label):
        nonlocal mirror_back
        mirror_back = not mirror_back
        update_display()
    backfill_toggle.on_clicked(on_backfill_clicked)

    scales, rotations = [], []
    yaws = []

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)

    # Final interpolation after the window is closed
    n_points = num_frames
    if n_points > 0:
        scales, rotations, yaws = interpolate_transformations(n_points)
        rotations = [-x for x in rotations]  # keep old sign convention
        path = as_numpy_array(path)
        path = as_numpy_array([linterp(path, i) for i in np.linspace(0, len(path) - 1, num=n_points)])

    return path, scales, rotations, yaws, mirror_back


def animate_polygon(image, polygon, path, scales, rotations, yaws=None, mirror_back=True, interp=cv2.INTER_LINEAR):
    """Animate selection with optional non-planar spin around a vertical axis through the pivot.
    If `yaws` is provided, uses a per-row surface-of-revolution re-sampling (not a cylinder);
    otherwise, falls back to the original planar affine transform.
    """
    frames = []
    transformed_polygons = []
    origin = np.array(path[0], dtype=np.float32)

    h, w = image.shape[:2]

    use_spin = yaws is not None

    if not use_spin:
        # Original planar affine path
        for i in eta(range(len(path)), title="Creating frames for this layer..."):
            theta = np.deg2rad(rotations[i])
            scale = float(scales[i])
            a11 = scale * np.cos(theta)
            a12 = -scale * np.sin(theta)
            a21 = scale * np.sin(theta)
            a22 = scale * np.cos(theta)
            tx = path[i][0] - (a11 * origin[0] + a12 * origin[1])
            ty = path[i][1] - (a21 * origin[0] + a22 * origin[1])
            M = np.array([[a11, a12, tx], [a21, a22, ty]])
            warped_image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            polygon_np = np.array(polygon)
            ones = np.ones(shape=(len(polygon_np), 1))
            points_ones = np.hstack([polygon_np, ones])
            transformed_polygon = M.dot(points_ones.T).T
            transformed_polygons.append(transformed_polygon)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(transformed_polygon)], 255)
            rgba_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2BGRA)
            alpha_channel = np.zeros((h, w), dtype=np.uint8)
            alpha_channel[mask == 255] = 255
            rgba_image[:, :, 3] = alpha_channel
            rgba_image[mask == 0] = (0, 0, 0, 0)
            frames.append(rgba_image)
        return EasyDict(frames=frames, transformed_polygons=transformed_polygons)

    # Spin mode: pre-extract polygon patch once
    poly_np = np.array(polygon, dtype=np.float32)
    x_min = max(0, int(np.floor(np.min(poly_np[:, 0]))))
    y_min = max(0, int(np.floor(np.min(poly_np[:, 1]))))
    x_max = min(w, int(np.ceil(np.max(poly_np[:, 0]))))
    y_max = min(h, int(np.ceil(np.max(poly_np[:, 1]))))
    if x_max <= x_min or y_max <= y_min:
        return EasyDict(frames=[], transformed_polygons=[])
    patch = image[y_min:y_max, x_min:x_max].copy()
    ph, pw = patch.shape[:2]
    patch_mask = np.zeros((ph, pw), dtype=np.uint8)
    shifted_poly = (poly_np - np.array([x_min, y_min], dtype=np.float32)).astype(np.float32)
    cv2.fillPoly(patch_mask, [shifted_poly.astype(np.int32)], 255)
    if patch.ndim == 3 and patch.shape[2] == 3:
        patch_rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    else:
        # For noise or other shapes
        c = patch.shape[2] if patch.ndim == 3 else 1
        arr = patch if c >= 3 else np.repeat(patch, 3, axis=2)
        arr = arr[:, :, :3].copy()
        patch_rgba = np.concatenate([arr, np.ones((ph, pw, 1), dtype=arr.dtype) * 255], axis=2)
    patch_rgba[:, :, 3] = patch_mask
    pivot_in_patch = origin - np.array([x_min, y_min], dtype=np.float32)
    # Precompute back-face once for speed
    Mflip = np.array([[-1, 0, 2 * float(pivot_in_patch[0])], [0, 1, 0]], dtype=np.float32)
    patch_back = cv2.warpAffine(
        patch_rgba,
        Mflip,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    is_noise_like = image.dtype != np.uint8

    for i in eta(range(len(path)), title="Creating frames for this layer..."):
        yaw = float(yaws[i]) if isinstance(yaws, (list, tuple, np.ndarray)) else float(yaws)
        roll = float(rotations[i]) if isinstance(rotations, (list, tuple, np.ndarray)) else float(rotations)
        scale = float(scales[i])

        spun = spin_patch(
            patch_rgba,
            pivot_x=float(pivot_in_patch[0]),
            yaw_deg=yaw,
            fov_deg=60.0,
            back_flipped_rgba=patch_back if mirror_back else None,
            fill_backface=mirror_back,
        )
        if abs(roll) > 1e-3:
            spun = rotate_patch_around_pivot(spun, pivot_in_patch, roll)
        if abs(scale - 1.0) > 1e-3:
            spun, pivot_scaled = scale_patch_about_pivot(spun, pivot_in_patch, scale)
        else:
            pivot_scaled = pivot_in_patch.copy()

        out_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        dx = int(round(path[i][0] - pivot_scaled[0]))
        dy = int(round(path[i][1] - pivot_scaled[1]))
        out_rgba = paste_rgba(out_rgba, spun, dx, dy)

        mask = out_rgba[:, :, 3]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            transformed_polygons.append(cnt.reshape(-1, 2))
        else:
            transformed_polygons.append(np.zeros((0, 2), dtype=np.float32))
        frames.append(out_rgba)

    return EasyDict(frames=frames, transformed_polygons=transformed_polygons)


def apply_transformation(polygon, scale, rotation, origin):
    # Translate polygon to origin
    translated_polygon = polygon - origin
    # Apply scaling
    scaled_polygon = translated_polygon * scale
    # Apply rotation
    theta = np.deg2rad(rotation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_polygon = np.dot(scaled_polygon, rotation_matrix)
    # Translate back
    final_polygon = rotated_polygon + origin
    return final_polygon


def spin_patch(patch_rgba, pivot_x, yaw_deg=0.0, fov_deg=60.0, back_flipped_rgba=None, fill_backface=True):
    """Spin an RGBA patch like a rigid, zero-thickness plane around a vertical axis through pivot_x.
    - yaw_deg in degrees; backside uses the same texture flipped about the pivot.
    - Projects to a trapezoid using a simple pinhole camera with FOV, then warps into the same HxW canvas.
    """
    H, W = patch_rgba.shape[:2]
    a = np.deg2rad(yaw_deg)

    # Wrap to [-90, 90] for shape; detect backface separately
    yaw_view = ((yaw_deg + 90.0) % 180.0) - 90.0
    av = np.deg2rad(yaw_view)
    backface = np.cos(a) < 0.0

    # Choose source (flip around pivot for backside)
    src = patch_rgba
    if backface:
        if not fill_backface:
            # No backside content
            src = np.zeros_like(patch_rgba)
        else:
            if back_flipped_rgba is not None:
                src = back_flipped_rgba
            else:
                M = np.array([[-1, 0, 2 * float(pivot_x)], [0, 1, 0]], dtype=np.float32)
                src = cv2.warpAffine(
                    patch_rgba,
                    M,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )

    # Focal length from FOV
    f = 0.5 * W / np.tan(np.deg2rad(max(1e-3, fov_deg)) / 2.0)

    def x_project(x):
        xr = x - float(pivot_x)
        x_rot = xr * np.cos(av)
        z = -xr * np.sin(av)
        # Simple pinhole, camera at z=f looking to z=0
        denom = (f - z)
        denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
        return float(pivot_x) + f * (x_rot / denom)

    # Source quad (rectangle)
    src_quad = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    # Destination quad: x projected, y unchanged
    xL = x_project(0)
    xR = x_project(W - 1)
    dst_quad = np.array([[xL, 0], [xR, 0], [xR, H - 1], [xL, H - 1]], dtype=np.float32)

    Hmat = cv2.getPerspectiveTransform(src_quad, dst_quad)
    out = cv2.warpPerspective(
        src,
        Hmat,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return out


def rotate_patch_around_pivot(patch_rgba, pivot_xy, angle_deg):
    """Rotate RGBA patch around pivot inside canvas size."""
    H, W = patch_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((float(pivot_xy[0]), float(pivot_xy[1])), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        patch_rgba,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def scale_patch_about_pivot(patch_rgba, pivot_xy, scale):
    """Scale RGBA patch about pivot; returns (scaled_patch, scaled_pivot)."""
    H, W = patch_rgba.shape[:2]
    new_W = max(1, int(round(W * scale)))
    new_H = max(1, int(round(H * scale)))
    scaled = cv2.resize(patch_rgba, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    pivot_scaled = np.array([pivot_xy[0] * scale, pivot_xy[1] * scale], dtype=np.float32)
    return scaled, pivot_scaled


def paste_rgba(dest_rgba, src_rgba, x, y):
    """Paste src RGBA onto dest RGBA at top-left (x, y) with alpha overwrite."""
    H, W = dest_rgba.shape[:2]
    h, w = src_rgba.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return dest_rgba
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    region_dst = dest_rgba[y0:y1, x0:x1]
    region_src = src_rgba[sy0:sy1, sx0:sx1]
    alpha = region_src[:, :, 3:4]
    mask = (alpha > 0)
    region_dst[mask[:, :, 0]] = region_src[mask[:, :, 0]]
    dest_rgba[y0:y1, x0:x1] = region_dst
    return dest_rgba


def compute_spin_geometry(alpha_mask, pivot_x):
    """Precompute per-row left/right/R for spin; faster across frames.
    alpha_mask: HxW uint8 or bool mask of the polygon within patch.
    """
    H, W = alpha_mask.shape[:2]
    left = np.full(H, -1, dtype=np.int32)
    right = np.full(H, -1, dtype=np.int32)
    for y in range(H):
        row = alpha_mask[y] > 0
        if not np.any(row):
            continue
        xs = np.where(row)[0]
        left[y] = xs[0]
        right[y] = xs[-1]
    R = np.zeros(H, dtype=np.float32)
    for y in range(H):
        l, r = left[y], right[y]
        if l < 0 or r < 0:
            R[y] = 0.0
            continue
        R[y] = max(float(pivot_x) - l, r - float(pivot_x))
        if R[y] < 0:
            R[y] = 0.0
    return dict(left=left, right=right, R=R)


# def cogvlm_caption_video(video_path, prompt="Please describe this video in detail."):
#     import rp.web_evaluator as wev
#
#     client = wev.Client("100.113.27.133")
#     result = client.evaluate("run_captioner(x,prompt=prompt)", x=video_path, prompt=prompt)
#     if result.errored:
#         raise result.error
#     return result.value


if __name__ == "__main__":
    fansi_print(big_ascii_text("Go With The Flow!"), "yellow green", "bold")

    image_path = input_conditional(
        fansi("First Frame: Enter Image Path or URL", "blue cyan", "italic bold underlined"),
        lambda x: is_a_file(x.strip()) or is_valid_url(x.strip()),
    ).strip()

    print("Using path: " + fansi_highlight_path(image_path))
    if is_video_file(image_path):
        fansi_print('Video path was given. Using first frame as image.')
        image=load_video(image_path,length=1)[0]
    else:
        image = load_image(image_path, use_cache=True)
        image = resize_image_to_fit(image, height=1440, allow_growth=False)

    rp.fansi_print("PRO TIP: Use this website to help write your captions: https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space", 'blue cyan')
    prompt=input(fansi('Input the video caption >>> ','blue cyan','bold'))

    SCALE_FACTOR=1
    #Adjust resolution to 720x480: resize then center-crop
    HEIGHT=480*SCALE_FACTOR
    WIDTH=720*SCALE_FACTOR
    image = resize_image_to_hold(image,height=HEIGHT,width=WIDTH) 
    image = crop_image(image, height=HEIGHT,width=WIDTH, origin='center')
    title = input_default(
        fansi("Enter a title: ", "blue cyan", "italic bold underlined"),
        get_file_name(
            image_path,
            include_file_extension=False,
        ),
    )
    output_folder=make_directory(get_unique_copy_path(title))
    print("Output folder: " + fansi_highlight_path(output_folder))

    fansi_print("How many layers?", "blue cyan", "italic bold underlined"),
    num_layers = input_integer(
        minimum=1,
    )

    layer_videos = []
    layer_polygons = []
    layer_first_frame_masks = []
    layer_noises = []

    for layer_num in range(num_layers):
        layer_noise=np.random.randn(HEIGHT,WIDTH,18).astype(np.float32)

        fansi_print(f'You are currently working on layer #{layer_num+1} of {num_layers}','yellow orange','bold')
        if True or not "polygon" in vars() or input_yes_no("New Polygon?"):
            polygon = select_polygon(image)
        if True or not "animation" in vars() or input_yes_no("New Animation?"):
            animation = select_path(image, polygon)

        
        animation_output = animate_polygon(image, polygon, *animation)

        noise_output_1 = as_numpy_array(animate_polygon(layer_noise[:,:,3*0:3*1], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_2 = as_numpy_array(animate_polygon(layer_noise[:,:,3*1:3*2], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_3 = as_numpy_array(animate_polygon(layer_noise[:,:,3*2:3*3], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_4 = as_numpy_array(animate_polygon(layer_noise[:,:,3*3:3*4], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_5 = as_numpy_array(animate_polygon(layer_noise[:,:,3*4:3*5], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_6 = as_numpy_array(animate_polygon(layer_noise[:,:,3*5:3*6], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_warp_output = np.concatenate(
            [
                noise_output_1[:,:,:,:3],
                noise_output_2[:,:,:,:3],
                noise_output_3[:,:,:,:3],
                noise_output_4[:,:,:,:3],
                noise_output_5[:,:,:,:3],
                noise_output_6[:,:,:,:1],
            ],
            axis=3,#THWC
        )

        frames, transformed_polygons = destructure(animation_output)

        mask = get_image_alpha(frames[0]) > 0
        
        layer_polygons.append(transformed_polygons)
        layer_first_frame_masks.append(mask)
        layer_videos.append(frames)
        layer_noises.append(noise_warp_output)

    if True or input_yes_no("Inpaint background?"):
        total_mask = sum(layer_first_frame_masks).astype(bool)
        background = cv_inpaint_image(image, mask=total_mask)
    else:
        background = "https://t3.ftcdn.net/jpg/02/76/96/64/360_F_276966430_HsEI96qrQyeO4wkcnXtGZOm0Qu4TKCgR.jpg"
        background = load_image(background, use_cache=True)
        background = cv_resize_image(background, get_image_dimensions(image))
        background=as_rgba_image(background)

    ###
    output_frames = [
        overlay_images(
            background,
            *frame_layers,
        )
        for frame_layers in eta(list_transpose(layer_videos),title=fansi("Compositing all frames of the video...",'green','bold'))
    ]
    output_frames=as_numpy_array(output_frames)

    
    output_video_file=save_video_mp4(output_frames, output_folder+'/'+title + ".mp4", video_bitrate="max")
    output_mask_file = save_video_mp4(
        [
            sum([get_image_alpha(x) for x in layers])
            for layers in list_transpose(layer_videos)
        ],
        output_folder + "/" + title + "_mask.mp4",
        video_bitrate="max",
    )
    

    ###
    fansi_print("Warping noise...",'yellow green','bold italic')
    output_noises = np.random.randn(1,HEIGHT,WIDTH,16)
    output_noises=np.repeat(output_noises,49,axis=0)
    for layer_num in range(num_layers):
        fansi_print(f'Warping noise for layer #{layer_num+1} of {num_layers}','green','bold')
        for frame in eta(range(49),title='frame number'):
            noise_mask = get_image_alpha(layer_videos[layer_num][frame])[:,:,None]>0
            noise_video_layer = layer_noises[layer_num][frame]
            output_noises[frame]*=(noise_mask==0)
            output_noises[frame]+=noise_video_layer*noise_mask
            #display_image((noise_mask * noise_video_layer)[:,:,:3])
            display_image(output_noises[frame][:,:,:3]/5+.5)
    
    import einops
    import torch
    torch_noises=torch.tensor(output_noises)
    torch_noises=einops.rearrange(torch_noises,'F H W C -> F C H W')        
    #
    small_torch_noises=[]
    for i in eta(range(49),title='Regaussianizing'):
        torch_noises[i]=nw.regaussianize(torch_noises[i])[0]
        small_torch_noise=nw.resize_noise(torch_noises[i],(480//8,720//8))
        small_torch_noises.append(small_torch_noise)
        #display_image(as_numpy_image(small_torch_noise[:3])/5+.5)
        display_image(as_numpy_image(torch_noises[i,:3])/5+.5)
    small_torch_noises=torch.stack(small_torch_noises)#DOWNSAMPLED NOISE FOR CARTRIDGE!

    ###
    cartridge={}
    cartridge['instance_noise']=small_torch_noises.bfloat16()
    cartridge['instance_video']=(as_torch_images(output_frames)*2-1).bfloat16()
    cartridge['instance_prompt']=prompt
    output_cartridge_file=object_to_file(cartridge, output_folder + "/" + title + "_cartridge.pkl")
            
    ###
    
    
    output_polygons_file=output_folder+'/'+'polygons.npy'
    polygons=as_numpy_array(layer_polygons)
    np.save(output_polygons_file,polygons)
    
    print()
    print(fansi('Saved outputs:','green','bold'))
    print(fansi('    - Saved video: ','green','bold'),fansi_highlight_path(get_relative_path(output_video_file)))
    print(fansi('    - Saved masks: ','green','bold'),fansi_highlight_path(get_relative_path(output_mask_file)))
    print(fansi('    - Saved shape: ','green','bold'),fansi_highlight_path(output_polygons_file))
    print(fansi('    - Saved cartridge: ','green','bold'),fansi_highlight_path(output_cartridge_file))

    print("Press CTRL+C to exit")


    display_video(video_with_progress_bar(output_frames), loop=True)
