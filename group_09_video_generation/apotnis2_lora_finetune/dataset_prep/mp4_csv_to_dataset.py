#!/usr/bin/env python3
"""
extract 6-second clips from *left_image*.mp4 files in bags folder
, generate captions using gemini 2.0 and add movement text
"""

import os
import time
from pathlib import Path
from moviepy import VideoFileClip
import google.generativeai as genai
import pandas as pd
from plot_odom import get_movement_descriptions

# configure gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def load_and_slice_dataframe(video_path: str, start_time: float, end_time: float):
    """load csv file and slice dataframe based on time range"""
    bags_dir = Path("bags")
    odom_files = list(bags_dir.glob("*odom.csv"))
    
    if not odom_files:
        return None
        
    try:
        df = pd.read_csv(odom_files[0])
        
        if 'timestamp_ns' not in df.columns:
            return None
        
        # convert and normalize timestamps
        df['timestamp_s'] = (df['timestamp_ns'] / 1e9) - (df['timestamp_ns'].iloc[0] / 1e9)
        
        # slice dataframe
        mask = (df['timestamp_s'] >= start_time) & (df['timestamp_s'] <= end_time)
        return df[mask].copy()
    except:
        return None


def generate_caption(video_path: str) -> str:
    """generate caption for video using gemini 2.0"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # upload video
    video_file = genai.upload_file(video_path)
    
    # wait for file to be active
    max_wait_time = 60
    wait_time = 0
    
    while video_file.state.name == "PROCESSING" and wait_time < max_wait_time:
        time.sleep(2)
        wait_time += 2
        
        # refresh file state
        try:
            video_file = genai.get_file(video_file.name)
        except:
            pass
    
    if video_file.state.name != "ACTIVE":
        # cleanup failed file
        try:
            genai.delete_file(video_file.name)
        except:
            pass
        raise Exception(f"file upload failed, final state: {video_file.state.name}")
    
    # generate caption
    response = model.generate_content([
        "This is a video recorded in the view point of a robot. describe this video in exactly 350 characters or less. be concise and descriptive about objects, movements etc and the scene. answer text only, no points or markdown..",
        video_file
    ])
    
    # cleanup
    genai.delete_file(video_file.name)
    
    return response.text.strip()[:400]


def create_clips():
    bags_dir = Path("bags")
    output_dir = Path("clips")
    output_dir.mkdir(exist_ok=True)
    
    # output files for consolidated data
    prompt_file = output_dir / "prompt.txt"
    videos_file = output_dir / "videos.txt"
    
    # find all left_image mp4 files
    videos = list(bags_dir.glob("*left_image*.mp4"))
    
    for video in videos:
        print(f"processing {video.name}")
        
        try:
            # load video
            clip = VideoFileClip(str(video))
            duration = clip.duration
            
            # create 6-second clips
            start = 0
            clip_num = 0
            
            while start < duration:
                end = min(start + 6, duration)
                clip_name = f"{video.stem}_clip_{clip_num:03d}.mp4"
                clip_path = output_dir / clip_name
                
                try:
                    # extract subclip
                    subclip = clip.subclipped(start, end)
                    subclip.write_videofile(str(clip_path))
                    subclip.close()
                    
                    # generate caption
                    caption = generate_caption(str(clip_path))
                    
                    # load and slice dataframe for this time range
                    sliced_df = load_and_slice_dataframe(str(video), start, end)
                    if sliced_df is not None and len(sliced_df) > 0:
                        movement_descriptions = " the robot goes " + get_movement_descriptions(sliced_df)
                    else:
                        movement_descriptions = ""
                    
                    # append caption to prompt.txt
                    with open(prompt_file, 'a') as f:
                        f.write(caption + movement_descriptions + '\n')
                    
                    # append video filename to videos.txt
                    with open(videos_file, 'a') as f:
                        f.write(clip_name + '\n')
                    
                    print(f"  created {clip_name}")
                except Exception as e:
                    print(f"  failed to create {clip_name}: {e}")
                
                start += 6
                clip_num += 1
            
            clip.close()
            
        except Exception as e:
            print(f"failed to process {video.name}: {e}")
    
    print(f"output files: {prompt_file}, {videos_file}")


if __name__ == "__main__":
    create_clips()