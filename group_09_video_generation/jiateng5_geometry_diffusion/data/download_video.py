#!/usr/bin/env python3
"""
Video Download Script for RealEstate10K Dataset

This script downloads YouTube videos from URLs found in .txt files
in the RealEstate10K dataset and saves them to organized directories.
"""

import os
import sys
import logging
import subprocess
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import cv2
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VideoDownloader:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.download_stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0
        }
        self.lock = threading.Lock()
        
    def extract_youtube_url(self, file_path: str) -> Optional[str]:
        """Extract YouTube URL from the first line of a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('https://www.youtube.com/watch?v='):
                    return first_line
                elif 'youtube.com' in first_line:
                    return first_line
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        return None
    
    def get_file_id(self, file_path: str) -> str:
        """Get the filename (without extension) as the unique identifier."""
        return Path(file_path).stem
    
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def is_video_downloaded(self, output_dir: str, file_id: str) -> bool:
        """Check if video is already downloaded."""
        video_files = [
            f"{file_id}.mp4",
            f"{file_id}.mkv",
            f"{file_id}.webm",
            f"{file_id}.avi"
        ]
        
        for video_file in video_files:
            if os.path.exists(os.path.join(output_dir, video_file)):
                return True
        return False
    
    def download_video(self, url: str, output_dir: str, file_id: str, video_id: str) -> bool:
        """Download a single video using yt-dlp."""
        try:
            # Check if video already exists
            if self.is_video_downloaded(output_dir, file_id):
                logger.info(f"Video {file_id} already exists, skipping...")
                with self.lock:
                    self.download_stats['skipped'] += 1
                return True
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # yt-dlp command (let yt-dlp choose the best format automatically)
            cmd = [
                'yt-dlp',
                '--output', f'{output_dir}/{file_id}.%(ext)s',
                '--no-playlist',
                '--no-write-info-json',  # Don't download JSON metadata
                '--no-write-thumbnail',  # Don't download thumbnail
                '--progress',  # Show progress bar
                '--newline',   # Better progress display
                url
            ]
            
            logger.info(f"Downloading video {file_id} (YouTube ID: {video_id})...")
            
            # Run yt-dlp with real-time output
            result = subprocess.run(
                cmd,
                text=True,
                timeout=1800  # 30 minute timeout for full videos
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded video {file_id}")
                with self.lock:
                    self.download_stats['successful'] += 1
                return True
            else:
                logger.error(f"Failed to download video {file_id} (exit code: {result.returncode})")
                with self.lock:
                    self.download_stats['failed'] += 1
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout downloading video {file_id}")
            with self.lock:
                self.download_stats['failed'] += 1
            return False
        except Exception as e:
            logger.error(f"Error downloading video {file_id}: {e}")
            with self.lock:
                self.download_stats['failed'] += 1
            return False
    
    def process_file(self, file_path: str, output_dir: str) -> bool:
        """Process a single .txt file and download its video."""
        try:
            url = self.extract_youtube_url(file_path)
            if not url:
                logger.warning(f"No YouTube URL found in {file_path}")
                return False
            
            video_id = self.get_video_id(url)
            if not video_id:
                logger.warning(f"Could not extract video ID from {url}")
                return False
            
            file_id = self.get_file_id(file_path)
            return self.download_video(url, output_dir, file_id, video_id)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def get_txt_files(self, directory: str) -> List[str]:
        """Get all .txt files in a directory."""
        txt_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.txt'):
                        txt_files.append(os.path.join(root, file))
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return txt_files
    
    def download_videos_from_directory(self, input_dir: str, output_dir: str):
        """Download all videos from .txt files in a directory."""
        logger.info(f"Processing directory: {input_dir}")
        
        txt_files = self.get_txt_files(input_dir)
        logger.info(f"Found {len(txt_files)} .txt files")
        
        if not txt_files:
            logger.warning(f"No .txt files found in {input_dir}")
            return
        
        # Update total count
        with self.lock:
            self.download_stats['total'] += len(txt_files)
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for txt_file in txt_files:
                future = executor.submit(self.process_file, txt_file, output_dir)
                futures.append(future)
            
            # Process completed downloads with progress tracking
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(txt_files):
                        logger.info(f"Progress: {completed}/{len(txt_files)} files processed")
                except Exception as e:
                    logger.error(f"Error in future: {e}")
                    completed += 1
    
    def download_and_process_videos_from_directory(self, input_dir: str, output_dir: str):
        """Download and process videos from all .txt files in a directory with enhanced processing."""
        logger.info(f"Processing directory: {input_dir}")
        
        txt_files = self.get_txt_files(input_dir)
        if not txt_files:
            logger.warning(f"No .txt files found in {input_dir}")
            return
        
        logger.info(f"Found {len(txt_files)} .txt files to process")
        
        # Create temp directory for downloads
        temp_dir = "./temp_downloads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process files with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file_path in txt_files:
                future = executor.submit(self.download_and_process_video, file_path, output_dir)
                futures.append(future)
                with self.lock:
                    self.download_stats['total'] += 1
            
            # Process completed downloads with progress tracking
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(txt_files):
                        logger.info(f"Progress: {completed}/{len(txt_files)} files processed")
                except Exception as e:
                    logger.error(f"Error in future: {e}")
                    completed += 1
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary download directory")
    
    def parse_annotation_file(self, annotation_path):
        """Parse the annotation file and extract pose data"""
        logger.info(f"Parsing annotation file: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # First line is the URL
        url = lines[0].strip()
        video_id = self.get_video_id(url)
        
        # Use the filename (without extension) as the unique identifier
        file_id = Path(annotation_path).stem
        
        # Parse pose data (skip first line which is URL)
        pose_data = []
        timestamps = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                values = line.strip().split()
                if len(values) == 19:  # timestamp + 18 pose parameters
                    # Store timestamp
                    timestamps.append(int(values[0]))
                    # Convert to float and remove timestamp (first value)
                    pose_row = [float(v) for v in values[1:]]  # Remove timestamp
                    pose_data.append(pose_row)
        
        pose_tensor = torch.tensor(pose_data, dtype=torch.float32)
        logger.info(f"Parsed {len(pose_data)} pose frames for file {file_id} (video: {video_id})")
        
        return file_id, video_id, url, pose_tensor, timestamps
    
    def extract_frames_from_video(self, video_path, output_dir, timestamps, target_fps=10.0):
        """Extract frames from video based on timestamps and save as new video at target FPS"""
        logger.info(f"Extracting frames from {video_path} based on timestamps")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        logger.info(f"Original video: {total_frames} frames at {original_fps} FPS, duration: {duration:.2f}s")
        
        # Convert timestamps to frame indices
        # Timestamps are in microseconds, convert to seconds
        first_timestamp = timestamps[0] / 1000000.0
        last_timestamp = timestamps[-1] / 1000000.0
        
        logger.info(f"Timestamp range: {first_timestamp:.2f}s to {last_timestamp:.2f}s")
        
        # Calculate frame indices based on timestamps
        frame_indices = []
        for timestamp in timestamps:
            # Convert timestamp to seconds, then to frame index
            time_seconds = timestamp / 1000000.0
            frame_idx = int(time_seconds * original_fps)
            if frame_idx < total_frames:
                frame_indices.append(frame_idx)
        
        logger.info(f"Will extract {len(frame_indices)} frames at {target_fps} FPS")
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, f"{Path(video_path).stem}.mp4")
        
        # Get frame dimensions from first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read first frame")
            cap.release()
            return None
        
        height, width, channels = frame.shape
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                logger.warning(f"Could not read frame {frame_idx}")
        
        cap.release()
        out.release()
        
        logger.info(f"Extracted {len(frame_indices)} frames to {output_path}")
        return output_path
    
    def create_pose_file(self, pose_tensor, output_dir, file_id):
        """Create pose file in the same format as the mini dataset"""
        pose_path = os.path.join(output_dir, f"{file_id}.pt")
        torch.save(pose_tensor, pose_path)
        logger.info(f"Saved pose data to {pose_path}")
        return pose_path
    
    def create_metadata_file(self, video_paths, pose_paths, output_dir):
        """Create metadata file in the same format as the mini dataset"""
        metadata = {
            'video_fps': torch.tensor([10.0] * len(video_paths)),
            'video_pts': [torch.arange(len(torch.load(pose_path))) * 1024 for pose_path in pose_paths],
            'video_paths': video_paths
        }
        
        metadata_path = os.path.join(output_dir, "metadata.pt")
        torch.save(metadata, metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    def download_and_process_video(self, file_path: str, output_dir: str) -> bool:
        """Download video and process it with pose data extraction"""
        try:
            # Parse annotation file
            file_id, video_id, url, pose_tensor, timestamps = self.parse_annotation_file(file_path)
            
            # Create subdirectories
            video_dir = os.path.join(output_dir, "test_256" if "test" in file_path else "train_256")
            pose_dir = os.path.join(output_dir, "test_poses" if "test" in file_path else "train_poses")
            
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            
            # Check if already processed
            video_path = os.path.join(video_dir, f"{file_id}.mp4")
            pose_path = os.path.join(pose_dir, f"{file_id}.pt")
            
            if os.path.exists(video_path) and os.path.exists(pose_path):
                logger.info(f"Video {file_id} already processed, skipping...")
                with self.lock:
                    self.download_stats['skipped'] += 1
                return True
            
            # Download video to temporary location
            temp_dir = "./temp_downloads"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download the full video first
            success = self.download_video(url, temp_dir, file_id, video_id)
            if not success:
                logger.error(f"Failed to download video {file_id}")
                return False
            
            # Find the downloaded video file
            video_files = list(Path(temp_dir).glob(f"{file_id}.*"))
            if not video_files:
                logger.error(f"No video file found for {file_id}")
                return False
            
            temp_video_path = str(video_files[0])
            
            # Extract frame range based on timestamps
            cropped_video_path = self.extract_frames_from_video(
                temp_video_path, 
                video_dir, 
                timestamps,
                target_fps=10.0
            )
            
            if not cropped_video_path:
                logger.error(f"Failed to extract frames for {file_id}")
                os.remove(temp_video_path)
                return False
            
            # Create pose file
            self.create_pose_file(pose_tensor, pose_dir, file_id)
            
            # Clean up temporary file
            os.remove(temp_video_path)
            
            logger.info(f"Successfully processed video {file_id}")
            with self.lock:
                self.download_stats['successful'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            with self.lock:
                self.download_stats['failed'] += 1
            return False
    
    def print_stats(self):
        """Print download statistics."""
        logger.info("=" * 50)
        logger.info("DOWNLOAD STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {self.download_stats['total']}")
        logger.info(f"Successfully downloaded: {self.download_stats['successful']}")
        logger.info(f"Failed downloads: {self.download_stats['failed']}")
        logger.info(f"Skipped (already exists): {self.download_stats['skipped']}")
        logger.info("=" * 50)

def main():
    """Main function to orchestrate the video download process."""
    
    # Define paths
    base_dir = "/shared/nas/data/m1/jiateng5/GeometryForcing/data/real-estate-10k"
    
    input_dirs = {
        "test": os.path.join(base_dir, "RealEstate10K", "test"),
        "train": os.path.join(base_dir, "RealEstate10K", "train")
    }
    
    output_dirs = {
        "test": os.path.join(base_dir, "test"),
        "train": os.path.join(base_dir, "train")
    }
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        logger.info("yt-dlp is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("yt-dlp is not installed. Please install it first:")
        logger.error("pip install yt-dlp")
        sys.exit(1)
    
    # Create output directories
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Initialize downloader
    downloader = VideoDownloader(max_workers=4)
    
    # Process each directory
    for split in ["train", "test"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING {split.upper()} SPLIT")
        logger.info(f"{'='*60}")
        
        input_dir = input_dirs[split]
        output_dir = output_dirs[split]
        
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            continue
        
        start_time = time.time()
        downloader.download_and_process_videos_from_directory(input_dir, output_dir)
        end_time = time.time()
        
        logger.info(f"Completed {split} split in {end_time - start_time:.2f} seconds")
    
    # Print final statistics
    downloader.print_stats()
    
    logger.info("Video download process completed!")

if __name__ == "__main__":
    main()
