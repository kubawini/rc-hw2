"""
Stub for homework 2
"""
import math
import time
import random
from enum import Enum

import numpy as np
import mujoco
from mujoco import viewer


import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 1


world_xml_path = f"car_{TASK_ID}.xml"
model = mujoco.MjModel.from_xml_path(world_xml_path)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(
    n_steps: int, /, view=True, rendering_speed = 10, **controls: float
) -> NDArray[np.uint8]:
    """A wrapper around `mujoco.mj_step` to advance the simulation held in
    the `data` and return a photo from the dash camera installed in the car.

    Args:
        n_steps: The number of simulation steps to take.
        view: Whether to render the simulation.
        rendering_speed: The speed of rendering. Higher values speed up the rendering.
        controls: A mapping of control names to their values.
        Note that the control names depend on the XML file.

    Returns:
        A photo from the dash camera at the end of the simulation steps.

    Examples:
        # Advance the simulation by 100 steps.
        sim_step(100)

        # Move the car forward by 0.1 units and advance the simulation by 100 steps.
        sim_step(100, **{"forward": 0.1})

        # Rotate the dash cam by 0.5 radians and advance the simulation by 100 steps.
        sim_step(100, **{"dash cam rotate": 0.5})
    """

    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img



# TODO: add addditional functions/classes for task 1 if needed
def rotate_car(img):
    height, width, _ = img.shape
    img_stripe_red = np.max(img[:, (width//2)-10 : (width//2)+10, 0])
    img_stripe_red_right = np.max(img[:, (width//2)+10 : (width//2)+20, 0])
    if np.any(img_stripe_red > 210):
        return img_stripe_red, {"turn": 0}
    elif np.any(img_stripe_red_right > 210):
        return img_stripe_red, {"turn": -0.05}
    else:
        return img_stripe_red, {"turn": 0.05}


def go_forward(turn, img):
    height, width, _ = img.shape
    img_stripe_red_sum = np.sum(np.logical_and(img[:,:,0] > 100, np.logical_and(img[:,:,1] < 50, img[:,:,2] < 50)))
    if img_stripe_red_sum > 7000 and turn == 0:
        return True, {'forward': 0}
    elif img_stripe_red_sum > 2000 and turn == 0:
        return False, {'forward': 0.1}
    elif turn == 0:
        return False, {'forward': 1}
    else:
        return False, {'forward': 0}
# /TODO


def task_1():
    steps = random.randint(0, 2000)
    controls = {"forward": 0, "turn": 0.5}
    img = sim_step(steps, view=False, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    while True:
        # print('car: ', data.body("car").xpos)
        # print('ball: ', data.body("target-ball").xpos)
        red, controls = rotate_car(img)
        end, forward = go_forward(controls['turn'], img)
        controls.update(forward)
        if(end):
            break
        img = sim_step(100, view=True, **controls)

    # /TODO



# TODO: add addditional functions/classes for task 2 if needed
class Stage(Enum):
    ONE = 0,
    TWO = 1,
    THREE = 2,
    FOUR = 3,
    FIVE = 4,
    SIX = 5,
    SEVEN = 6,
    EIGHT = 7,
    NINE = 8,
    TEN = 9

def move_to_wall(i):
    if i == 10:
        return Stage.TWO, {'turn': 0, 'forward': 0}, i+1
    else:
        return Stage.ONE, {'turn': 0, 'forward': 1}, i+1

def move_a_little_backwards(i):
    if i == 10:
        return Stage.THREE, {'turn': 0, 'forward': 0}, i+1
    else:
        return Stage.TWO, {'turn': 0, 'forward': -0.5}, i+1


def localize_pole(img):
    height, width, _ = img.shape
    middle = width // 2
    diff_arr = np.array(img[10:, middle, :], dtype=int) - np.array(img[:-10, middle, :], dtype=int)
    diff_arr = diff_arr.reshape(470,-1,3)
    diff_arr[diff_arr < 0] = 0
    threshold = np.array([80, 80, 80])
    condition = np.all(diff_arr > threshold, axis=(1,2))
    if np.any(condition == True):
        return Stage.FOUR, {'turn': 0, 'forward': 0}
    else:
        return Stage.THREE, {'turn': 0.1, 'forward': 0}


def rotate_towards_blue_wall(img, stage_beg, stage_end):
    height, width, _ = img.shape
    calibration_index = int(18 / 30 * width)
    right_img_red = np.array(img[:, calibration_index, 0], dtype=np.int_)
    right_img_green = np.array(img[:, calibration_index, 1], dtype=np.int_)
    right_img_blue = np.array(img[:, calibration_index, 2], dtype=np.int_)
    right_img_red_blue = (right_img_red + right_img_blue)
    if not (np.any(right_img_green > right_img_red_blue)):
        return stage_end, {'turn': 0, 'forward': 0}
    else:
        return stage_beg, {'turn': 0.1, 'forward': 0}


def move_a_little_forward(i, max_iter, stage_beg, stage_end):
    if i == max_iter:
        return stage_end, {'turn': 0, 'forward': 0}, i+1
    else:
        return stage_beg, {'turn': 0, 'forward': 1}, i+1


def rotate_towards_green_wall(img):
    height, width, _ = img.shape
    calibration_index = int(18 / 30 * width)
    right_img_red = np.array(img[:, calibration_index, 0], dtype=np.int_)
    right_img_green = np.array(img[:, calibration_index, 1], dtype=np.int_)
    right_img_blue = np.array(img[:, calibration_index, 2], dtype=np.int_)
    right_img_red_green = (right_img_red + right_img_green)
    if not (np.any(right_img_blue > right_img_red_green)):
        return Stage.SEVEN, {'turn': 0, 'forward': 0}
    else:
        return Stage.SIX, {'turn': 0.1, 'forward': 0}


# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=True, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    current_stage = Stage.ONE
    s1 = 0
    s2 = 0
    s5 = 0
    s7 = 0
    s9 = 0
    while True:
        # print(data.body("car").xpos)
        # print(data.body("target-ball").xpos)
        match current_stage:
            case Stage.ONE:
                current_stage, controls, s1 = move_to_wall(s1)
            case Stage.TWO:
                current_stage, controls, s2 = move_a_little_backwards(s2)
            case Stage.THREE:
                current_stage, controls = localize_pole(img)
            case Stage.FOUR:
                current_stage, controls = rotate_towards_blue_wall(img, Stage.FOUR, Stage.FIVE)
            case Stage.FIVE:
                current_stage, controls, s5 = move_a_little_forward(s5, 50, Stage.FIVE, Stage.SIX)
            case Stage.SIX:
                current_stage, controls = rotate_towards_green_wall(img)
            case Stage.SEVEN:
                current_stage, controls, s7 = move_a_little_forward(s7, 50, Stage.SEVEN, Stage.EIGHT)
            case Stage.EIGHT:
                current_stage, controls = rotate_towards_blue_wall(img, Stage.EIGHT, Stage.NINE)
            case Stage.NINE:
                current_stage, controls, s9 = move_a_little_forward(s9, 50, Stage.NINE, Stage.TEN)
            case Stage.TEN:
                red, controls = rotate_car(img)
                end, forward = go_forward(controls['turn'], img)
                controls.update(forward)
                if (end):
                    break

        img = sim_step(20, view=True, **controls)

    # /TODO



def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
    return np.linalg.norm(ball_pos - car_pos) < 0.2


def ball_grab() -> bool:
    """Checks if the ball is inside the gripper."""
    print(data.body("target-ball").xpos[2])
    return data.body("target-ball").xpos[2] > 0.1


def teleport_by(x: float, y: float) -> None:
    data.qpos[0] += x
    data.qpos[1] += y
    sim_step(10, **{"dash cam rotate": 0})


def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    fovy = 90
    f = h / (2 * np.tan(fovy * np.pi / 360))
    intrinsic_matrix = np.array([[-f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients


# TODO: add addditional functions/classes for task 3 if needed
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def find_teleport_coordinates(tvecs, ids):
    x_d = 0
    y_d = 0
    for tvec, id in zip(tvecs, ids):
        if id == 1:
            tvec_flatten = tvec.reshape(-1)
            radius = math.sqrt(tvec_flatten[0]**2 + tvec_flatten[2]**2)
            view_angle = math.atan(tvec_flatten[0] / tvec_flatten[2])
            angle = math.pi / 4 + view_angle
            x_d = math.cos(angle) * radius
            y_d = math.sin(angle) * radius
    x_update = 1 + 0.3 - x_d - 0.1
    y_update = 2 + 0.25 - y_d
    return x_update, y_update
# /TODO


def task_3():
    start_x = random.uniform(-0.2, 0.2)
    start_y = random.uniform(0, 0.2)
    teleport_by(start_x, start_y)

    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function

    controls = {'dash cam rotate': -0.1}
    img = sim_step(750, view=True, **controls)

    controls = {'trapdoor close/open': 1}
    sim_step(1000, view=True, **controls)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detectorParams = cv2.aruco.DetectorParameters()
    detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

    MARKER_SIDE = 0.08

    corners, ids, _ = detector.detectMarkers(img)
    img_draw = img.copy()
    cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)

    camera_matrix, dist_coeffs = get_dash_camera_intrinsics()
    rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, MARKER_SIDE, camera_matrix, dist_coeffs)

    time.sleep(2)
    x_dest, y_dest = find_teleport_coordinates(tvecs, ids)

    teleport_by(x_dest, y_dest)
    time.sleep(2)

    # /TODO

    assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can move the car as well if you need to
    controls = {'lift': 1}
    sim_step(1000, view=True, **controls)
    controls = {'forward': -0.1}
    sim_step(10, view=True, **controls)
    controls = {'trapdoor close/open': -1}
    sim_step(1000, view=True, **controls)
    controls = {'lift': -1}
    sim_step(1000, view=True, **controls)
    controls = {'trapdoor close/open': 1}
    sim_step(1000, view=True, **controls)
    controls = {'lift': 1}
    sim_step(1000, view=True, **controls)
    # /TODO

    assert ball_grab()


if __name__ == "__main__":
    print(f"Running TASK_ID {TASK_ID}")
    if TASK_ID == 1:
        task_1()
    elif TASK_ID == 2:
        task_2()
    elif TASK_ID == 3:
        task_3()
    else:
        raise ValueError("Unknown TASK_ID")
