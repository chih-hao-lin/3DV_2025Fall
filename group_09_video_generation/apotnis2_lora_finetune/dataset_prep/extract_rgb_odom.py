#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import cv2
from rosbags.highlevel import AnyReader

# topic names
IMAGE_TOPIC = '/terrasentia/zed2/zed_node/left/image_rect_color/compressed'
ODOM_TOPIC = '/terrasentia/ekf'

# message types 
COMPRESSED_IMAGE_TYPE = 'sensor_msgs/msg/CompressedImage'
ODOM_TYPE = 'nav_msgs/msg/Odometry'


def decode_compressed_image(msg):
    """convert compressed image to bgr frame"""
    buf = np.frombuffer(memoryview(msg.data).toreadonly(), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('failed to decode compressed image')
    
    # ensure bgr format
    if frame.shape[2] == 4:  # bgra -> bgr
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 3:  # already bgr
        pass
    else:
        raise ValueError(f'unexpected image channels: {frame.shape[2]}')
    
    return frame


def extract_odom_data(msg):
    """extract odometry data from message (ros2)"""
    timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
    
    position = [
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ]
    
    orientation = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    
    linear_velocity = [
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z
    ]
    
    angular_velocity = [
        msg.twist.twist.angular.x,
        msg.twist.twist.angular.y,
        msg.twist.twist.angular.z
    ]
    
    return {
        'timestamp': timestamp,
        'position': position,
        'orientation': orientation,
        'linear_velocity': linear_velocity,
        'angular_velocity': angular_velocity
    }


def infer_fps(timestamps, sample=200):
    """estimate fps from timestamps"""
    if len(timestamps) < 2:
        return 30.0
    
    diffs = []
    for i in range(1, min(sample, len(timestamps))):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            diffs.append(dt)
    
    if not diffs:
        return 30.0
    
    avg_dt_ns = sum(diffs) / len(diffs)
    return float(1e9 / avg_dt_ns)


def process_bag(bag_path):
    """process single bag file"""
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f'bag file not found: {bag_path}')
    
    # output files
    bag_name = bag_path.stem
    mp4_file = bag_path.parent / f'{bag_name}_zed_left.mp4'
    csv_file = bag_path.parent / f'{bag_name}_odom.csv'
    
    with AnyReader([bag_path]) as reader:
        # find connections
        image_conn = None
        odom_conn = None
        
        print("searching for topics...")
        for conn in reader.connections:
            # print(f"  found: {conn.topic} - {conn.msgtype}")
            if conn.topic == IMAGE_TOPIC:
                if conn.msgtype == COMPRESSED_IMAGE_TYPE:
                    image_conn = conn
                    print(f"found image topic: {conn.topic}")
            elif conn.topic == ODOM_TOPIC:
                if conn.msgtype == ODOM_TYPE:
                    odom_conn = conn
                    print(f"found odom topic: {conn.topic}")
                
        if not image_conn:
            print("image topic missing or wrong type")
        if not odom_conn:
            print("odom topic missing or wrong type")
            
        # extract image data
        print(f'extracting images from {IMAGE_TOPIC}...')
        timestamps = []
        frames = []
        
        for _, ts, raw in reader.messages(connections=[image_conn]):
            msg = reader.deserialize(raw, image_conn.msgtype)
            try:
                frame = decode_compressed_image(msg)
                frames.append(frame)
                timestamps.append(ts)
            except Exception as e:
                print(f'warning: failed to decode frame: {e}')
                continue
        
        if not frames:
            raise RuntimeError('no frames extracted')
        
        # write video (bgr format)
        fps = infer_fps(timestamps)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(str(mp4_file), fourcc, fps, (w, h), isColor=True)
        if not writer.isOpened():
            raise RuntimeError(f'failed to open video writer for {mp4_file}')
        
        for frame in frames:
            # frames are already bgr format
            writer.write(frame)
        
        writer.release()
        print(f'video saved: {mp4_file} ({len(frames)} frames, {fps:.1f} fps)')
        
        # extract odom data
        print(f'extracting odometry from {ODOM_TOPIC}...')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # header
            writer.writerow([
                'timestamp_ns',
                'position_x', 'position_y', 'position_z',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
            ])
            
            message_count = 0
            
            for _, ts, raw in reader.messages(connections=[odom_conn]):
                msg = reader.deserialize(raw, odom_conn.msgtype)
                try:
                    odom_data = extract_odom_data(msg)
                    writer.writerow([
                        odom_data['timestamp'],
                        *odom_data['position'],
                        *odom_data['orientation'],
                        *odom_data['linear_velocity'],
                        *odom_data['angular_velocity']
                    ])
                    message_count += 1
                except Exception as e:
                    print(f'warning: failed to process odom message: {e}')
                    continue
        
        print(f'odometry saved: {csv_file} ({message_count} messages)')


def main():
    ap = argparse.ArgumentParser(description='extract zed left camera and odometry from bag file')
    ap.add_argument('bag_file', help='path to bag file')
    args = ap.parse_args()
    
    try:
        process_bag(args.bag_file)
        print('extraction complete')
    except Exception as e:
        print(f'error: {e}')
    


if __name__ == '__main__':
    main()
