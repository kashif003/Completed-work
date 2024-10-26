import cv2
import numpy as np
import pyniryo
from pyniryo2 import Vision, NiryoRos
import time
# Define the positions for the Tic Tac Toe field and turn indicators
positions = {
    "p1": (247, 154),
    "p2": (319, 148),
    "p3": (394, 148),
    "p4": (252, 229),
    "p5": (325, 216),
    "p6": (400, 213),
    "p7": (255, 300),
    "p8": (333, 295),
    "p9": (411, 288)
}

indications = {
    "I1": (465, 155),  # Indication point for player 1
    "I2": (482, 260)   # Indication point for player 2
}

# Corrected HSV color ranges for red, green, and blue
color_ranges = {
    'red': ([0, 100, 100], [10, 255, 255]),  # Lower half of red hue (0-10)
    'green': ([35, 50, 50], [85, 255, 255]),  # Green (35-85)
    'blue': ([100, 150, 0], [140, 255, 255])   # Blue (100-140)
}

# Colors for drawing rectangles
draw_colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}

# Initialize the Niryo robot's ROS instance and vision system
ros_instance = NiryoRos("169.254.200.200")
vision = Vision(ros_instance)

# Function to detect color in a given position using surrounding pixels
def detect_color(hsv_frame, pos):
    x, y = pos

    # Define offsets for the surrounding pixels (1 center and 8 surrounding)
    offsets = [
        (0, 0),     # Center pixel
        (-1, 0),    # Left
        (1, 0),     # Right
        (0, -1),    # Top
        (0, 1),     # Bottom
        (-1, -1),   # Top-left
        (1, -1),    # Top-right
        (-1, 1),    # Bottom-left
        (1, 1)      # Bottom-right
    ]

    # Dictionary to count color occurrences
    color_counts = {color: 0 for color in color_ranges.keys()}
    color_counts['none'] = 0  # Initialize 'none' count

    # Check colors of the surrounding pixels
    for dx, dy in offsets:
        x_offset = x + dx
        y_offset = y + dy
        
        # Ensure we don't go out of bounds
        if 0 <= x_offset < hsv_frame.shape[1] and 0 <= y_offset < hsv_frame.shape[0]:
            pixel_color = hsv_frame[y_offset, x_offset]

            # Check against predefined color ranges
            for color_name, (lower, upper) in color_ranges.items():
                lower_bound = np.array(lower, dtype=np.uint8)
                upper_bound = np.array(upper, dtype=np.uint8)

                # Check if the pixel color is within the color range
                if all(lower_bound <= pixel_color) and all(pixel_color <= upper_bound):
                    color_counts[color_name] += 1
                    break  # Stop checking once we find a match
            else:
                color_counts['none'] += 1  # If no color matched

    # Determine the most frequent color detected
    detected_color = max(color_counts, key=color_counts.get)
    return detected_color

# Initialize a state dictionary to store current colors of each position
state = {pos_name: "none" for pos_name in positions.keys()}
last_printed_green_positions = []

def positions_played():
    global last_printed_green_positions
    
    while True:
        # Get the camera frame
        img_compressed = vision.get_img_compressed()
        img_uncompressed = pyniryo.uncompress_image(img_compressed)
        camera_info = vision.get_camera_intrinsics()
        img = pyniryo.undistort_image(img_uncompressed, camera_info.intrinsics, camera_info.distortion)
        frame = cv2.resize(img, (640, 480))

        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect color at indication point
        i2_color = detect_color(hsv_frame, indications["I2"])
        if i2_color == "red":
        	
            current_green_positions = []
            time.sleep(3)
            
            for pos_name, pos in positions.items():
                # Detect color using the integrated function
                detected_color = detect_color(hsv_frame, pos)
                
                if detected_color != state[pos_name]:
                    state[pos_name] = detected_color

                if detected_color == "green":
                    current_green_positions.append(pos_name)

            # If the positions have changed, print the new positions
            if current_green_positions != last_printed_green_positions:
                print(f"Positions with green: {current_green_positions}")
                new_position = list(set(current_green_positions) - set(last_printed_green_positions))
                last_printed_green_positions = current_green_positions.copy()

                if not len(new_position) < 1:
                    return new_position

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
