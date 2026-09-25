"""
Minimal guided example: pick up a camera with the robot, place it at one or
more table x/y coordinates, capture beam images.

Self-contained: does NOT import other project .py files. Still needs:
  - installed packages: xarm, pyrealsense2, opencv-python, numpy, scipy
  - stereo_calibration2.npz in the working directory
  - reachable xArm at ARM_IP

This script does NOT control any mirror motors. It is only for:
  1. finding the camera by ArUco tag,
  2. picking it up and placing it on the table at (x_position_f, y_position_f),
  3. taking a beam image with the ELP USB camera (OpenCV / DirectShow).

Placement is an integrated locate -> grab flow (no home in between):
  - same approach as read_initial_positions / find_pos to get exact (x_i, y_i),
  - skip gohome(),
  - compute x_adjust from that exact pose, then grab and place at final x/y.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI


# ---------------------------------------------------------------------------
# Robot / vision setup (inlined from analyzer_laser_align.py)
# ---------------------------------------------------------------------------

ARM_IP = os.environ.get("ROBOT_ARM_IP", "").strip()
if not ARM_IP:
    raise RuntimeError("Set ROBOT_ARM_IP before importing the robot hardware skeleton.")
# The lab session may override this before initialize_cams().
CALIBRATION_PATH = Path("stereo_calibration2.npz")

arm = XArmAPI(ARM_IP)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
print("init done")


def start() -> None:
    try:
        arm.set_counter_reset()
        arm.set_tcp_load(
            weight=0.610,
            center_of_gravity=(0.06125, 0.0458, 0.0375),
        )
        arm.set_servo_angle(
            angle=[180, 75, -180, 20, 0, 90, -60],
            is_radian=False,
            speed=30,
            wait=True,
        )
    except Exception as e:
        print(f"MainException: {e}")


def initialize_cams():
    calibration_data = np.load(CALIBRATION_PATH)
    mtx1 = calibration_data["mtx1"]
    dist1 = calibration_data["dist1"]
    mtx2 = calibration_data["mtx2"]
    dist2 = calibration_data["dist2"]
    R_stereo = calibration_data["R"]
    T = calibration_data["T"]
    actual_distance = 0.23  # 23 cm
    calibrated_distance = 0.22  # 22 cm

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)

    scale_factor = actual_distance / calibrated_distance
    T = T * scale_factor
    _proj1 = mtx1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    _proj2 = mtx2 @ np.hstack((R_stereo, T))

    cap1 = cv2.VideoCapture(5, cv2.CAP_DSHOW)  # Same overhead camera as MSMF 0.
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap2 = cv2.VideoCapture(4, cv2.CAP_DSHOW)  # Door camera; same as MSMF 6.
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap3 = cv2.VideoCapture(2, cv2.CAP_MSMF)
    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 6000)
    print("cameras setup")
    return (cap1, cap2, cap3, pipeline)


def detect_aruco(image, target_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        if target_id in ids:
            index = np.where(ids == target_id)[0][0]
            return corners[index][0], int(ids[index])
    return None, None


def estimate_pose(corners, mtx, dist, marker_length):
    obj_points = np.array(
        [
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0],
        ]
    )
    _success, rvec, tvec = cv2.solvePnP(obj_points, corners, mtx, dist)
    return rvec, tvec


def calculate_rotation_angle(corner):
    center_x = (corner[0][0] + corner[1][0] + corner[2][0] + corner[3][0]) / 4
    center_y = (corner[0][1] + corner[1][1] + corner[2][1] + corner[3][1]) / 4
    midpoint_x = (corner[0][0] + corner[1][0]) / 2
    midpoint_y = (corner[0][1] + corner[1][1]) / 2
    vector_x = midpoint_x - center_x
    vector_y = midpoint_y - center_y
    return float(np.degrees(np.arctan2(vector_y, vector_x)))


def fine_adjust(arm_api, pipeline, target_id, timeout_s=None):
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    leave = False
    depth_image = None
    corners = None
    center_x = 0
    center_y = 0
    place = None
    while not leave:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("RealSense localization timed out; check the declared camera location.")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        corners, tag_id = detect_aruco(color_image, target_id)
        if tag_id is not None:
            center_x = int(
                (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4
            )
            center_y = int(
                (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4
            )
            movey = (320 - center_x) / 10
            movex = (240 - center_y) / 10
            if movex == 0 and movey == 0:
                leave = True
            _code, place = arm_api.get_position_aa(is_radian=False)
            arm_api.set_position_aa(
                [place[0] + movex] + [place[1] + movey] + place[2:],
                speed=20,
                mvacc=30,
                wait=True,
            )

    depth_value = depth_image[center_y, center_x]
    h, w = depth_image.shape
    center_h = h // 2
    center_w = w // 2
    square = depth_image[center_h - 1 : center_h + 2, center_w - 1 : center_w + 2]
    depth_value = np.mean(square)
    print(depth_value, "depth")

    rotation_angle = calculate_rotation_angle(corners) + 90
    arm_api.set_position_aa(
        [place[0] + 74.2] + [place[1] + 35.5] + place[2:],
        speed=50,
        mvacc=100,
        wait=True,
    )
    _code, pos = arm_api.get_servo_angle(servo_id=7, is_radian=False)
    arm_api.set_servo_angle(
        servo_id=7, wait=True, angle=pos + rotation_angle, is_radian=False
    )
    return depth_value, rotation_angle


def find_beam_center(
    img_input,
    use_otsu=True,
    relative_min_area=0.0001,
    fixed_threshold_val=None,
    blur_kernel=(11, 11),
):
    if img_input is None or img_input.size == 0:
        return None, None

    if img_input.dtype != np.uint8:
        max_val = np.max(img_input)
        if max_val > 0:
            img_input_8bit = (img_input.astype(np.float32) / max_val * 255).astype(
                np.uint8
            )
        else:
            img_input_8bit = np.zeros_like(img_input, dtype=np.uint8)
    else:
        img_input_8bit = img_input

    noise_floor = 50
    img_blur = cv2.GaussianBlur(img_input_8bit, blur_kernel, 0)
    flag_meaningful_signal = True

    if use_otsu:
        threshold_val, mask = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(threshold_val)
        if threshold_val < noise_floor:
            flag_meaningful_signal = False
    elif fixed_threshold_val is not None:
        threshold_val, mask = cv2.threshold(
            img_blur, fixed_threshold_val, 255, cv2.THRESH_BINARY
        )
    else:
        threshold_val = 100
        _, mask = cv2.threshold(img_blur, threshold_val, 255, cv2.THRESH_BINARY)

    total_area = img_input_8bit.shape[0] * img_input_8bit.shape[1]
    min_area_val = total_area * relative_min_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area_val and area > max_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                best_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                max_area = area

    return best_center, flag_meaningful_signal, mask


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# ArUco tag attached to the portable camera mount that the robot will pick up.
CAMERA_TAG = 16

# Physical size of that marker in meters.
CAMERA_MARKER_SIZE_M = 0.062

# Folder for saved beam snapshots.
SNAPSHOT_DIR = Path("snapshots_pick_and_place")

# ELP USB48MP02-CFV (OpenCV). Change INDEX if enumerate shows a different device.
# Must NOT be an index already held open by initialize_cams() (0, 2, 5).
ELP_INDEX = 2
ELP_BACKEND = cv2.CAP_DSHOW  # try cv2.CAP_MSMF if DSHOW fails
ELP_WIDTH = 8000
ELP_HEIGHT = 6000
ELP_AUTO_EXPOSURE = 0.25  # Windows DSHOW: often 0.25; try 0 or 1 if ignored
ELP_N_SKIP = 5  # discard frames after open/exposure so AE settles

# Gripper fine-offset calibration (same numbers as analyzer_laser_align_standalone).
X_UPPER, Y_UPPER = 358.41687 + 1.5, 88.222145
X_LOWER, Y_LOWER = 379.8, -345.884674
CALIB_Y = -4 / (Y_UPPER - Y_LOWER) * 0.75
CALIB_X = -0.01


@dataclass(frozen=True)
class CaptureStep:
    """
    One pick/place + one beam capture.

    Only x_position_f and y_position_f are required from the user. Placement
    height and tool orientation are handled inside the pick helper.

    exposure is the OpenCV CAP_PROP_EXPOSURE value (DSHOW scale; more negative
    is typically darker), not milliseconds.
    """

    name: str
    x_position_f: float
    y_position_f: float
    added_angle: float = 0.0
    exposure: float = -6.0
    settle_s: float = 0.2


# Replace these with your real camera-view x/y positions.
CAPTURE_SEQUENCE: list[CaptureStep] = [
    CaptureStep(
        name="view_1",
        x_position_f=201.8,
        y_position_f=-334.4,
        added_angle=0.0,
        exposure=-6.0,
    ),
    CaptureStep(
            name="view_2",
            x_position_f=204.8,
            y_position_f=-215.6,
            added_angle=0.0,
            exposure=-6.0,
        ),
    CaptureStep(
            name="view_3",
            x_position_f=283.5,
            y_position_f=-134.3,
            added_angle=-90.0,
            exposure=-6.0,
        ),
]


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def _x_adjust(x: float, y: float) -> float:
    """Same gripper offset calibration used by the standalone analyzer."""
    return (y - Y_LOWER) * CALIB_Y + (x - X_LOWER) * CALIB_X


def _stereo_mem_for_tag(
    vision_cams,
    tag_id: int,
    size_m: float = CAMERA_MARKER_SIZE_M,
    timeout_s: float = 2.0,
) -> list[float]:
    """
    Stereo detect tag -> mem pose list, same conversion as find_position_of_tag.
    Arm may nudge servo 1 if the tag is not seen; it does not approach yet.
    """
    cap1, cap2, _cap3, _pipeline = vision_cams
    calibration_data = np.load(CALIBRATION_PATH)
    mtx1 = calibration_data["mtx1"]
    dist1 = calibration_data["dist1"]
    mtx2 = calibration_data["mtx2"]
    dist2 = calibration_data["dist2"]

    arm.set_gripper_enable(True)
    arm.set_gripper_speed(2000)
    arm.set_gripper_position(850, wait=True)

    start_time = time.time()
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            raise RuntimeError("Could not read frame from one or both vision cameras.")
        preview_start = time.time()
        image_dir = Path("runs/localization_images") / str(time.time_ns())
        image_dir.mkdir(parents=True)
        for name, frame in (("cap1_mtx1.png", frame1), ("cap2_mtx2.png", frame2)):
            if not cv2.imwrite(str(image_dir / name), frame):
                raise RuntimeError(f"Could not save {image_dir / name}")
        print(f"Localization images saved to: {image_dir.resolve()}")
        input("Inspect the saved images, then press Enter to continue (Ctrl+C to abort): ")
        start_time += time.time() - preview_start  # Reviewing images is not a detection timeout.
        if time.time() - start_time > timeout_s:
            print("Timeout: No ArUco tag detected. Adjusting servo angle.")
            _code, angle = arm.get_servo_angle(servo_id=1, is_radian=False)
            arm.set_servo_angle(
                servo_id=1, angle=angle + 5, wait=True, is_radian=False, speed=100
            )
            start_time = time.time()

        corners1, _id1 = detect_aruco(frame1, tag_id)
        corners2, _id2 = detect_aruco(frame2, tag_id)
        if corners1 is None or corners2 is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Locate aborted by user.")
            continue

        _rvec1, tvec1 = estimate_pose(corners1, mtx1, dist1, size_m)
        _rvec2, tvec2 = estimate_pose(corners2, mtx2, dist2, size_m)
        avg_tvec = np.mean([tvec1, tvec2], axis=0)

        # Same as find_position_of_tag before find_pos (z fixed to 500 for approach).
        mem = avg_tvec.flatten().tolist()
        mem.extend([-180, 0, 0])
        mem[0] = -mem[0] * 1000
        mem[1] = mem[1] * 1000
        mem[2] = 500
        return mem


def find_exact_position_no_home(
    vision_cams,
    tag_id: int,
    size_m: float = CAMERA_MARKER_SIZE_M,
) -> tuple[list[float], float, float]:
    """
    Partial read_initial_positions / find_position_of_tag / find_pos:

      stereo -> approach above tag -> fine_adjust -> read exact TCP (x_i, y_i)

    Same as stock find_pos, but does NOT call gohome().
    Returns (place, rotation, depth_mm).
    """
    _cap1, _cap2, _cap3, pipeline = vision_cams
    mem = _stereo_mem_for_tag(vision_cams, tag_id, size_m=size_m)
    print(mem, "this is mem (stereo, before exact find_pos)")

    arm.set_gripper_enable(True)
    arm.set_gripper_speed(2000)
    arm.set_gripper_position(850, wait=True)

    speeds = 80
    x, y = mem[0], mem[1]
    new_angle = math.atan2(y, x) / math.pi * 180 + 180
    if x < 0 and y >= 0:
        new_angle = 280

    highcoor = [mem[0] - 70, mem[1] - 35, 500] + mem[3:]
    arm.set_servo_angle(servo_id=1, angle=new_angle, wait=True, is_radian=False, speed=speeds)
    arm.set_position_aa(highcoor, speed=speeds, mvacc=100, wait=True)

    depth, rotation = fine_adjust(arm, pipeline, tag_id)
    _code, place = arm.get_position_aa(is_radian=False)
    print(place, "exact place after fine_adjust (no gohome)")
    return list(place), float(rotation), float(depth)


def grab_and_place_from_located_pose(
    place: list[float],
    depth_mm: float,
    x_adj: float,
    final_x: float,
    final_y: float,
    added_angle: float = 0.0,
    *, arm_api=None,
) -> None:
    """
    Continue from the exact find_pos pose: apply x_adj, grasp, place at final x/y.

    Mirrors the grasp/drop half of pickup_claw_adjust_optimized_drop_w_angle,
    without re-approaching from home.
    """
    arm = arm_api if arm_api is not None else globals()["arm"]
    speeds = 80

    # fine_adjust already applied the nominal 74.2 / 35.5 offset; only add calib.
    arm.set_position_aa(
        [place[0] + x_adj, place[1], place[2]] + list(place[3:]),
        speed=50,
        mvacc=100,
        wait=True,
    )

    # Same height math as pickup_claw_adjust_optimized_drop_w_angle (non-special).
    height = depth_mm - 100
    height = 500 - height
    height += 16
    print(height, "grasp height")

    _code, place = arm.get_position_aa(is_radian=False)
    arm.set_position_aa(
        list(place[:2]) + [height] + list(place[3:]), speed=speeds, mvacc=100, wait=True
    )
    arm.set_gripper_position(520, wait=True)

    euler_angles = [180, 0, 90 + added_angle]
    r_inverse = R.from_euler("xyz", euler_angles, degrees=True)
    inverted_place = -r_inverse.as_rotvec() * 180 / np.pi
    target_angle = np.array([-inverted_place[0], -inverted_place[1], 0.0])

    arm.set_tcp_load(weight=0.8, center_of_gravity=(0.06125, 0.0458, 0.0375))
    arm.set_position_aa(
        [place[0], place[1], 580, target_angle[0], target_angle[1], target_angle[2]],
        speed=100,
        mvacc=100,
        wait=True,
    )

    drop_position = [final_x, final_y, height, target_angle[0], target_angle[1], target_angle[2]]
    drop_position_high = [final_x, final_y, 570, target_angle[0], target_angle[1], target_angle[2]]
    drop_position_middle = [
        final_x,
        final_y,
        height + 1.5,
        target_angle[0],
        target_angle[1],
        target_angle[2],
    ]

    arm.set_gripper_speed(1000)
    arm.set_position_aa(drop_position_high, is_radian=False, speed=100, mvacc=100, wait=True)
    arm.set_position_aa(drop_position_middle, is_radian=False, speed=80, mvacc=100, wait=True)
    arm.set_position_aa(drop_position, is_radian=False, speed=20, mvacc=100, wait=True)
    arm.set_gripper_position(850, wait=True)
    arm.set_tcp_load(weight=0.8, center_of_gravity=(0.06125, 0.0458, 0.0375))
    arm.set_gripper_speed(2000)
    arm.set_position_aa(drop_position_high, is_radian=False, speed=100, mvacc=100, wait=True)
    arm.set_servo_angle(
        angle=[180, 75, -180, 20, 0, 90, -60], speed=80, is_radian=False, wait=True
    )


def initialize_realsense_only():
    """Manual mode: no overhead cameras, beam camera, or stereo calibration."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)
    return (None, None, None, pipeline)


def place_camera_from_known_station(vision_cams, source, step):
    """Approximate RealSense approach from declared placement TCP coordinates.

    Fixed offsets use the original localization orientation. They are an initial
    viewing estimate, not a calibrated transform for arbitrary mount geometry.
    Optional per-station viewing coordinates override that estimate.
    """
    x = float(source.get("view_x", source["x_position_f"] - 74.2))
    y = float(source.get("view_y", source["y_position_f"] - 35.5))
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(2000)
    arm.set_gripper_position(850, wait=True)
    angle = math.atan2(y, x) / math.pi * 180 + 180
    if x < 0 and y >= 0:
        angle = 280
    print(f"RealSense viewing approach: x={x:.2f}, y={y:.2f}, z=500 mm")
    arm.set_servo_angle(servo_id=1, angle=angle, wait=True, is_radian=False, speed=80)
    arm.set_position_aa([x, y, 500, -180, 0, 0], speed=80, mvacc=100, wait=True)
    depth, _rotation = fine_adjust(arm, vision_cams[3], CAMERA_TAG, timeout_s=30)
    _code, place = arm.get_position_aa(is_radian=False)
    grab_and_place_from_located_pose(
        list(place), float(depth), _x_adjust(float(place[0]), float(place[1])),
        step.x_position_f, step.y_position_f, step.added_angle)


def place_camera_at_xy(vision_cams, step: CaptureStep) -> None:
    """
    Integrated pick/place using exact coordinates from the read_initial_positions path:

      1. find_exact_position_no_home  (stereo + approach + fine_adjust, no gohome)
      2. x_adj from that exact (x_i, y_i)
      3. grab_and_place_from_located_pose -> grasp and drop at final x/y
    """
    print(f"\n=== Place camera for {step.name} (tag={CAMERA_TAG}) ===")
    place, _rotation, depth_mm = find_exact_position_no_home(
        vision_cams, CAMERA_TAG, size_m=CAMERA_MARKER_SIZE_M
    )
    x_i, y_i = float(place[0]), float(place[1])
    x_adj = _x_adjust(x_i, y_i)

    print(
        f"  exact ({x_i:.2f}, {y_i:.2f}) -> "
        f"({step.x_position_f:.2f}, {step.y_position_f:.2f})  "
        f"[x_adj={x_adj:.3f}, added_angle={step.added_angle}]"
    )

    grab_and_place_from_located_pose(
        place,
        depth_mm,
        x_adj,
        step.x_position_f,
        step.y_position_f,
        step.added_angle,
    )


# ---------------------------------------------------------------------------
# Capture helper (ELP USB via OpenCV)
# ---------------------------------------------------------------------------

def open_elp_camera() -> cv2.VideoCapture:
    """Open the ELP at configured index/resolution and disable auto-exposure."""
    cap = cv2.VideoCapture(ELP_INDEX, ELP_BACKEND)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open ELP at index {ELP_INDEX} (backend={ELP_BACKEND}). "
            "Pick another ELP_INDEX or try CAP_MSMF."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ELP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ELP_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ELP_AUTO_EXPOSURE)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[elp] opened index={ELP_INDEX}  reported size={w}x{h}")
    return cap


def capture_beam_image(cap: cv2.VideoCapture, step: CaptureStep) -> np.ndarray | None:
    """Capture and save one beam image from the ELP USB camera."""
    out = SNAPSHOT_DIR / f"{step.name}.png"

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ELP_AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE, step.exposure)

    frame = None
    for _ in range(ELP_N_SKIP + 1):
        ok, frame = cap.read()
        if not ok:
            frame = None

    if frame is None:
        print(f"[capture] failed for {step.name}")
        return None

    cv2.imwrite(str(out), frame)
    print(
        f"[capture] saved {out}  shape={frame.shape}  "
        f"exposure={step.exposure} (prop={cap.get(cv2.CAP_PROP_EXPOSURE)})"
    )
    return frame


def print_beam_center_if_found(frame: np.ndarray, label: str) -> None:
    """Optional helper: analyze the saved image and print the beam center."""
    try:
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = frame

        result = find_beam_center(gray)
        if result is None or result[0] is None:
            print(f"[analysis] {label} beam center not found")
            return
        params, ok, _mask = result
        if ok and params is not None:
            print(f"[analysis] {label} beam center ~ ({params[0]:.1f}, {params[1]:.1f})")
        else:
            print(f"[analysis] {label} beam center not found")
    except (cv2.error, TypeError, ValueError) as e:
        print(f"[analysis] skipped for {label}: {e}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_capture_sequence(sequence: list[CaptureStep] = CAPTURE_SEQUENCE) -> None:
    """
    Typical usage:
      1. Start robot and vision cameras.
      2. Open the ELP USB camera.
      3. For each configured x/y:
           - exact locate (find_pos path, no gohome) then grab/place
           - capture one beam image from the ELP
      4. Release devices.

    After the last step, the camera remains on the table at the last placed x/y.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[start] {datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')}")
    vision_cams = None
    elp = None

    try:
        start()
        vision_cams = initialize_cams()
        elp = open_elp_camera()

        for step in sequence:
            print(f"\n=== {step.name} ===")
            place_camera_at_xy(vision_cams, step)

            time.sleep(step.settle_s)

            frame = capture_beam_image(elp, step)
            if frame is not None:
                print_beam_center_if_found(frame, step.name)

        print("\n=== Done ===")
        print(f"Saved images under: {SNAPSHOT_DIR}")
        print("Camera left at the last placed table coordinate.")

    finally:
        if elp is not None:
            elp.release()

        if vision_cams is not None:
            try:
                cap1, cap2, cap3, pipeline = vision_cams
                for cap in (cap1, cap2, cap3):
                    cap.release()
                pipeline.stop()
            except (AttributeError, RuntimeError) as e:
                print(f"[cleanup] failed to release vision cameras: {e}")


if __name__ == "__main__":
    run_capture_sequence()
    
