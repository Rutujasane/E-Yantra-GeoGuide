# -*- coding: utf-8 -*-
"""
This script controls a robot for an autonomous navigation challenge.

It performs the following key functions:
1.  Captures and processes a video feed of an arena.
2.  Uses ArUco markers for perspective transformation.
3.  Crops images of predefined "event" locations on the arena.
4.  Uses a pre-trained PyTorch model to classify the events.
5.  Determines a priority-based path for the robot to visit events.
6.  Tracks the robot's position using an ArUco marker.
7.  Sends movement commands to an ESP32 over Wi-Fi.
8.  Updates a CSV with the robot's current location.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import requests

# --- IMPORTANT: USER CONFIGURATION ---
# Please update the values below to match your specific hardware setup.

# Network Configuration: The IP address of your ESP32 Access Point.
# This is usually "192.168.4.1" by default.
ESP32_IP = "192.168.4.1"

# Camera Configuration: The index of your camera.
# This is often 0 for a built-in laptop camera or 1 for a USB webcam.
CAMERA_INDEX = 1

# --- System Constants (usually do not need to be changed) ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- END OF USER CONFIGURATION ---

# Image Processing Configuration
OUTPUT_IMAGE_SIZE = (1000, 1000)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
NODE_PROXIMITY_THRESHOLD = 70

# File Paths
MODEL_PATH = "best_model.pth"
LAT_LON_CSV_PATH = 'lat_long.csv'
LOCATION_UPDATE_CSV_PATH = "update_location.csv"
INITIAL_ARENA_CAPTURE_PATH = "arena_capture.png"
CROPPED_ARENA_PATH = "arena_cropped.png"
LIVE_ARENA_FEED_PATH = "live_arena_feed.png"

# Arena Layout: ArUco IDs for the four corners
ARENA_CORNER_IDS = {
    'top_left': 5, 'top_right': 4,
    'bottom_right': 6, 'bottom_left': 7
}

# Coordinates for event zones: (top_left, ..., bottom_left)
EVENT_ZONES = [
    ((186, 869), (287, 869), (287, 978), (186, 978)),  # Zone A
    ((658, 671), (761, 671), (761, 776), (658, 776)),  # Zone B
    ((666, 464), (768, 464), (768, 569), (666, 569)),  # Zone C
    ((171, 461), (271, 461), (271, 565), (171, 565)),  # Zone D
    ((188, 121), (290, 121), (290, 226), (188, 226)),  # Zone E
]

# Mapping of zone labels and navigation nodes to pixel coordinates
DESTINATIONS = {
    "S": (53, 918), "A": (232, 823), "B": (703, 619), "C": (708, 428),
    "D": (214, 411), "E": (235, 92),
    1: (104, 824), 2: (483, 823), 3: (94, 631), 4: (486, 619),
    5: (887, 624), 6: (92, 404), 7: (484, 417), 8: (889, 432),
    9: (92, 248), 10: (489, 268), 11: (891, 282)
}


# --- HELPER FUNCTIONS ---

def capture_initial_frame(cap):
    """Captures and saves a single frame from the camera."""
    result, frame = cap.read()
    if not result:
        print("Error: Could not read frame for initial capture.")
        return None
    cv2.imwrite(INITIAL_ARENA_CAPTURE_PATH, frame)
    return frame


def detect_aruco_details(img):
    """
    Detects ArUco markers in an image.

    Args:
        img: The image to process.

    Returns:
        A tuple of dictionaries: (marker_centers, marker_corners).
    """
    centers, corners_dict = {}, {}
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != 100:  # Bot marker handled separately
                centers[marker_id] = np.mean(corners[i][0], axis=0).astype(int)
                corners_dict[marker_id] = corners[i].reshape(4, 2)
    return centers, corners_dict


def apply_perspective_transform(image, aruco_centers):
    """
    Applies a perspective transform for a top-down view of the arena.

    Args:
        image: The source image.
        aruco_centers: A dictionary of detected ArUco marker centers.

    Returns:
        The transformed (warped) image, or None on failure.
    """
    try:
        source_pts = np.array([
            aruco_centers[ARENA_CORNER_IDS['top_left']],
            aruco_centers[ARENA_CORNER_IDS['top_right']],
            aruco_centers[ARENA_CORNER_IDS['bottom_right']],
            aruco_centers[ARENA_CORNER_IDS['bottom_left']],
        ], dtype=np.float32)

        output_pts = np.array([
            [0, 0], [OUTPUT_IMAGE_SIZE[0], 0],
            [OUTPUT_IMAGE_SIZE[0], OUTPUT_IMAGE_SIZE[1]],
            [0, OUTPUT_IMAGE_SIZE[1]]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(source_pts, output_pts)
        return cv2.warpPerspective(image, matrix, OUTPUT_IMAGE_SIZE)
    except KeyError as e:
        print(f"Error: Missing ArUco marker for transform: {e}")
        return None


# --- EVENT CLASSIFICATION FUNCTIONS ---

def _load_model(model_path):
    """Loads the pre-trained InceptionV3 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model.aux_logits = False

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def _classify_single_image(model, device, image_path):
    """Classifies a single event image using the loaded model."""
    classes = (
        'combat', 'destroyed_buildings', 'fire',
        'humanitarian_aid', 'military_vehicles'
    )
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = data_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
        _, predicted_idx = torch.max(prediction, 1)
    return classes[predicted_idx.item()]


def classify_events_on_arena(arena_image_path):
    """
    Crops event zones, classifies them, and determines visit order.

    Args:
        arena_image_path: Path to the cropped top-down arena view.

    Returns:
        A list of destination keys sorted by priority.
    """
    arena_image = cv2.imread(arena_image_path)
    classifications = {}
    model, device = _load_model(MODEL_PATH)
    zone_labels = ["A", "B", "C", "D", "E"]

    for i, coords in enumerate(EVENT_ZONES):
        vertices = np.array(coords, dtype=np.int32)
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)

        crop = arena_image[min_y:max_y, min_x:max_x]
        filename = f"event_{zone_labels[i]}.jpg"
        cv2.imwrite(filename, crop)

        # Simple check for empty (dark) event zones
        if np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < 10:
            classifications[zone_labels[i]] = "NO EVENT"
        else:
            classifications[zone_labels[i]] = _classify_single_image(
                model, device, filename
            )

    # --- Visualization ---
    label_map = {
        "fire": "Fire", "destroyed_buildings": "Destroyed",
        "combat": "Combat", "humanitarian_aid": "Aid",
        "military_vehicles": "Vehicles", "NO EVENT": "-"
    }
    for i, label in enumerate(zone_labels):
        display_name = label_map.get(classifications[label], "-")
        x, y = EVENT_ZONES[i][0]
        cv2.rectangle(arena_image, EVENT_ZONES[i][0],
                      EVENT_ZONES[i][2], (0, 255, 0), 2)
        cv2.putText(arena_image, f"{label}: {display_name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Event Classification', arena_image)
    cv2.waitKey(1)

    # --- Priority Sorting ---
    priority_order = {
        'Fire': 5, 'Destroyed': 4, 'Aid': 3,
        'Vehicles': 2, 'Combat': 1, '-': 0
    }
    sorted_zones = sorted(
        classifications.keys(),
        key=lambda z: priority_order.get(label_map.get(classifications[z]), 0),
        reverse=True
    )

    path = ['S'] + [
        z for z in sorted_zones if classifications[z] != "NO EVENT"
    ] + ['S']
    print(f"Identified Events: {classifications}")
    print(f"Calculated Path: {path}")
    return path


# --- BOT NAVIGATION AND CONTROL ---

def get_bot_location(live_arena_image):
    """Finds the bot's ArUco marker (ID 100) in the live feed."""
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)
    corners, ids, _ = detector.detectMarkers(live_arena_image)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 100:
                return np.mean(corners[i][0], axis=0)
    return None


def send_wifi_command(command):
    """Sends a numerical command to the ESP32 server."""
    url = f"http://{ESP32_IP}/?command={command}"
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            print(f"Command '{command}' sent successfully to ESP32.")
        else:
            print(f"Failed to send command. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending command to ESP32: {e}")


def get_bot_orientation(prev_node, curr_node):
    """
    Determines bot orientation and maps movement commands.
    Returns: A tuple (up, down, left, right) of bot action codes.
    """
    # Bot actions: 0=Fwd, 1=Left, 2=Right, 3=Bwd
    # Determine based on movement between last two nodes
    dx = curr_node[0] - prev_node[0]
    dy = curr_node[1] - prev_node[1]

    if abs(dx) < abs(dy):  # Primarily vertical movement
        if dy < 0:   # Moved UP on arena (face_up)
            return (0, 3, 1, 2)  # up=fwd, down=bwd, left=left, right=right
        else:        # Moved DOWN on arena (face_down)
            return (3, 0, 2, 1)  # up=bwd, down=fwd, left=right, right=left
    else:  # Primarily horizontal movement
        if dx > 0:  # Moved RIGHT on arena (face_right)
            return (2, 1, 0, 3)  # up=right, down=left, left=fwd, right=bwd
        else:       # Moved LEFT on arena (face_left)
            return (1, 2, 3, 0)  # up=left, down=right, left=bwd, right=fwd


def main():
    """Main execution function."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # --- Initial Setup ---
    print("Capturing initial arena image for setup...")
    initial_frame = capture_initial_frame(cap)
    if initial_frame is None:
        cap.release()
        return

    aruco_centers, _ = detect_aruco_details(initial_frame)
    if not all(k in aruco_centers for k in ARENA_CORNER_IDS.values()):
        print("Error: Not all corner markers detected. Check camera.")
        cap.release()
        return

    cropped_arena = apply_perspective_transform(initial_frame, aruco_centers)
    cv2.imwrite(CROPPED_ARENA_PATH, cropped_arena)

    # --- Event Classification ---
    print("Classifying events on the arena...")
    sorted_path = classify_events_on_arena(cropped_arena)

    # --- Bot Movement Loop ---
    print("\n--- Starting Bot Navigation ---")
    send_wifi_command(6)  # Command for "Start"
    path_index = 1
    location_history = [DESTINATIONS["S"]]  # Start at 'S'

    while path_index < len(sorted_path):
        ret, frame = cap.read()
        if not ret:
            continue

        live_arena = apply_perspective_transform(frame, aruco_centers)
        if live_arena is None:
            continue
        cv2.imshow("Live Arena Feed", live_arena)

        current_loc = get_bot_location(live_arena)
        if current_loc is None:
            continue

        target_key = sorted_path[path_index]
        target_loc = DESTINATIONS[target_key]

        # Check if target is reached
        distance_to_target = np.linalg.norm(current_loc - np.array(target_loc))
        if distance_to_target < 40:
            print(f"--- Reached Destination: {target_key} ---")
            send_wifi_command(5)  # Stop
            path_index += 1
            if path_index >= len(sorted_path):
                print("Final destination reached. Mission complete.")
                send_wifi_command(9)  # Victory
                break
            else:
                print(f"Next target: {sorted_path[path_index]}")
                cv2.waitKey(1000)  # Pause to stabilize

        # Check if bot is at any node to make a turning decision
        for node_id, node_coords in DESTINATIONS.items():
            # Only check numbered grid nodes for navigation decisions
            if not isinstance(node_id, int):
                continue

            dist_to_node = np.linalg.norm(current_loc - np.array(node_coords))
            if dist_to_node < NODE_PROXIMITY_THRESHOLD:
                # Avoid reprocessing the same node immediately
                last_node_dist = np.linalg.norm(
                    np.array(location_history[-1]) - np.array(node_coords)
                )
                if last_node_dist > 10:
                    location_history.append(node_coords)
                    prev_node = location_history[-2]
                    curr_node = location_history[-1]
                    print(f"At node {node_id}. Deciding next move.")

                    # Determine orientation and send command
                    up, down, left, right = get_bot_orientation(
                        prev_node, curr_node
                    )
                    # Simplified navigation logic
                    if abs(target_loc[1] - curr_node[1]) > 35:
                        send_wifi_command(up if target_loc[1] < curr_node[1]
                                          else down)
                    elif abs(target_loc[0] - curr_node[0]) > 35:
                        send_wifi_command(left if target_loc[0] < curr_node[0]
                                          else right)
                    break  # Exit node check loop after finding one

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual exit requested.")
            send_wifi_command(5)  # Stop bot
            break

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
