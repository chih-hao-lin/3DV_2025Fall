#!/usr/bin/env python3
"""
plot ekf odometry x,y positions from csv file and function to convert odom to text for movement descriptions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# constants
DIRECTION_CHANGE_THRESHOLD = -0.2  # m/s threshold for direction change
ANGLE_CHANGE_THRESHOLD = 7  # degrees threshold for angle change
LINEAR_MERGE_WINDOW = 3.0  # merge window for linear direction changes
ANGULAR_MERGE_WINDOW = 3.0  # merge window for angular changes
STOPPED_DISTANCE_THRESHOLD = 0.01  # meters threshold for stopped movement
TURN_DETECTION_THRESHOLD = 5  # degrees threshold for turn detection in descriptions

BAGS_DIR = "bags"


def quaternion_to_yaw(qx, qy, qz, qw):
    """convert quaternion to yaw angle in degrees"""
    yaw_rad = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.degrees(yaw_rad)

def normalize_angle(angle_diff):
    """normalize angle difference to [-180, 180]"""
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff

def get_timestamps(df):
    """get normalized timestamps in seconds"""
    if 'timestamp_ns' not in df.columns:
        return None
    return (df['timestamp_ns'] / 1e9 - df['timestamp_ns'].iloc[0] / 1e9)

def combine_change_points(linear_change_pts, angular_change_pts):
    """combine and sort all change points"""
    all_changes = [(pt, 'linear') for pt in linear_change_pts] + [(pt, 'angle') for pt in angular_change_pts]
    return sorted(all_changes, key=lambda x: x[0])

def merge_noisy_detections(change_pts, angle_changes, timestamps, merge_window=1.0):
    """merge multiple noisy detections within a time window"""
    
    if not change_pts or timestamps is None:
        return change_pts, angle_changes
    
    merged_pts = []
    merged_angles = []
    i = 0
    
    while i < len(change_pts):
        merge_start_time = timestamps.iloc[change_pts[i]]
        total_angle = angle_changes[i]
        last_pt = change_pts[i]
        
        # accumulate all points within merge window
        j = i + 1
        while j < len(change_pts):
            next_pt = change_pts[j]
            if timestamps.iloc[next_pt] - merge_start_time <= merge_window:
                total_angle += angle_changes[j]
                last_pt = next_pt
                j += 1
            else:
                break
        
        merged_pts.append(last_pt)
        merged_angles.append(total_angle)
        i = j  # skip to next unprocessed point
    
    return merged_pts, merged_angles

def dirn_change_pts(df):
    """detect direction change points based on linear velocity and angular changes"""
    
    # get linear velocity magnitude (forward/backward speed)
    linear_vel = df['linear_velocity_x']  # assuming x is forward direction
    
    # direction states: < DIRECTION_CHANGE_THRESHOLD m/s reverse, else forward or stop
    directions = ['reverse' if vel < DIRECTION_CHANGE_THRESHOLD else 'forward' for vel in linear_vel]
    
    # find direction change points (linear velocity changes)
    linear_change_pts = []
    for i in range(1, len(directions)):
        if directions[i] != directions[i-1]:
            linear_change_pts.append(i)
    
    # merge multiple linear direction changes within 3-second window
    timestamps = get_timestamps(df)
    if timestamps is not None and linear_change_pts:
        linear_change_pts, _ = merge_noisy_detections(linear_change_pts, [0] * len(linear_change_pts), timestamps, merge_window=LINEAR_MERGE_WINDOW)
    
    # convert quaternion to yaw angles
    yaw_angles = [quaternion_to_yaw(row['orientation_x'], row['orientation_y'], 
                                   row['orientation_z'], row['orientation_w']) 
                  for _, row in df.iterrows()]
    
    # find angular change points (rotation > 7 degrees)
    angular_change_pts = []
    angle_changes = []
    for i in range(1, len(yaw_angles)):
        angle_diff = yaw_angles[i] - yaw_angles[i-1]
        
        # normalize angle difference to [-180, 180]
        angle_diff = normalize_angle(angle_diff)
            
        if abs(angle_diff) > ANGLE_CHANGE_THRESHOLD:  # angle change threshold
            angular_change_pts.append(i)
            angle_changes.append(angle_diff)
    
    # merge multiple turn clips within 3-second window
    if timestamps is not None and angular_change_pts:
        angular_change_pts, angle_changes = merge_noisy_detections(angular_change_pts, angle_changes, timestamps, merge_window=ANGULAR_MERGE_WINDOW)
    
    return linear_change_pts, angular_change_pts, directions, yaw_angles, angle_changes

def create_movement_description(df, prev_idx, curr_idx, yaw_angles):
    """create movement description between two indices"""
    
    # get position data
    x, y = df['position_x'], df['position_y']
    
    # calculate relative coordinates between points
    prev_x, prev_y = x.iloc[prev_idx], y.iloc[prev_idx]
    curr_x, curr_y = x.iloc[curr_idx], y.iloc[curr_idx]
    
    # relative coordinates (delta x, delta y)
    delta_x = curr_x - prev_x
    delta_y = curr_y - prev_y
    
    # distance
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    # angle change
    prev_angle = yaw_angles[prev_idx]
    curr_angle = yaw_angles[curr_idx]
    angle_diff = curr_angle - prev_angle
    angle_diff = normalize_angle(angle_diff)  # normalize to [-180, 180]
    
    # determine direction based on velocity
    if prev_idx < len(df):
        vel = df['linear_velocity_x'].iloc[prev_idx]
        direction = "forward" if vel >= DIRECTION_CHANGE_THRESHOLD else "backward"
    else:
        direction = "forward"
    
    # create description
    if distance < STOPPED_DISTANCE_THRESHOLD:
        description = f"{direction} (stopped)"
    elif abs(angle_diff) > TURN_DETECTION_THRESHOLD:
        turn_dir = "left" if angle_diff > 0 else "right"
        description = f"{direction} {distance:.1f}metres {abs(angle_diff):.0f}degree {turn_dir}"
    else:
        description = f"{direction} {distance:.1f}metres"
    
    return description, delta_x, delta_y, distance, angle_diff

def get_movement_descriptions(df):
    """get movement descriptions as a single string joined by 'then'"""
    
    # detect direction change points
    linear_change_pts, angular_change_pts, directions, yaw_angles, angle_changes = dirn_change_pts(df)
    
    # combine and sort all change points
    all_changes = combine_change_points(linear_change_pts, angular_change_pts)
    
    descriptions = []
    prev_idx = 0
    
    for change_idx, _ in all_changes:
        description, _, _, _, _ = create_movement_description(df, prev_idx, change_idx, yaw_angles)
        descriptions.append(description)
        prev_idx = change_idx
    
    # final segment to end
    if prev_idx < len(df) - 1:
        description, _, _, _, _ = create_movement_description(df, prev_idx, len(df) - 1, yaw_angles)
        descriptions.append(description)
    
    return " then ".join(descriptions)

def print_change_points(linear_change_pts, angular_change_pts, timestamps):
    """print change points and indices in combined order"""
    
    # combine and sort all change points
    all_changes = combine_change_points(linear_change_pts, angular_change_pts)
    
    print("\n=== Change Points ===")
    print(f"{'Index':<6} {'Type':<8} {'Time':<8}")
    print("-" * 25)
    
    for change_idx, change_type in all_changes:
        time_str = f"{timestamps.iloc[change_idx]:.1f}" if timestamps is not None else "N/A"
        print(f"{change_idx:<6} {change_type:<8} {time_str:<8}")

def plot_odom(df):
    """detect and print change points and movement steps, then plot trajectory"""
    
    # detect direction change points
    linear_change_pts, angular_change_pts, directions, yaw_angles, angle_changes = dirn_change_pts(df)
    
    # get timestamps
    timestamps = get_timestamps(df)
    
    # print change points
    print_change_points(linear_change_pts, angular_change_pts, timestamps)
    
    # plot trajectory
    plt.figure(figsize=(12, 8))
    
    # plot trajectory line
    plt.plot(df['position_x'], df['position_y'], 'b-', alpha=0.7, linewidth=1, label='trajectory')
    
    # plot start point
    plt.plot(df['position_x'].iloc[0], df['position_y'].iloc[0], 'go', markersize=8, label='start')
    
    # plot end point
    plt.plot(df['position_x'].iloc[-1], df['position_y'].iloc[-1], 'ro', markersize=8, label='end')
    
    # plot linear change points
    if linear_change_pts:
        linear_x = df['position_x'].iloc[linear_change_pts]
        linear_y = df['position_y'].iloc[linear_change_pts]
        plt.plot(linear_x, linear_y, 'rs', markersize=6, label=f'linear changes ({len(linear_change_pts)})')
    
    # plot angular change points
    if angular_change_pts:
        angular_x = df['position_x'].iloc[angular_change_pts]
        angular_y = df['position_y'].iloc[angular_change_pts]
        plt.plot(angular_x, angular_y, 'yo', markersize=6, label=f'angular changes ({len(angular_change_pts)})')
    
    plt.xlabel('position x (m)')
    plt.ylabel('position y (m)')
    plt.title('odometry trajectory with change points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # find odom csv files
    bags_dir = Path(BAGS_DIR)
    odom_files = list(bags_dir.glob("*odom.csv"))
    
    if not odom_files:
        print("no odom csv files found in bags/")
    else:
        for csv_file in odom_files:
            print(f"processing {csv_file.name}")
            df = pd.read_csv(csv_file)
            # uncommet to only keep first 60 seconds of data relative to start time
            # df['timestamp_s'] = df['timestamp_ns'] / 1e9
            # start_time = df['timestamp_s'].iloc[0]
            # df = df[df['timestamp_s'] <= start_time + 60.0]
            plot_odom(df)

            print(get_movement_descriptions(df))
