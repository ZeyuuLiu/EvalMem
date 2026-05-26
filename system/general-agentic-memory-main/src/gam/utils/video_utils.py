
import time
import sys
import os
import base64
import numpy as np
import threading
import queue
from typing import List, Dict
import re
import os
import time
import queue
import threading
import json

try:
    import cv2
except ImportError:
    cv2 = None
    print("OpenCV not found!")


def get_frame_indices(video_path, start=0, end=None, n_frames=None, fps=None, max_frames=None):
    """
    Get frame indices based on either n_frames or fps.
    """
    if (n_frames is None and fps is None) or (n_frames is not None and fps is not None):
         raise ValueError("Either n_frames or fps must be provided, but not both.")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = int(start * video_fps)
    end_frame = int(end * video_fps) if end is not None else total_frames - 1
    end_frame = min(end_frame, total_frames - 1)
    
    if start_frame >= end_frame:
        return np.array([], dtype=int)

    if n_frames is not None:
        indices = np.linspace(start_frame, end_frame, n_frames).astype(int)
    else:
        # fps mode
        # Calculate step size
        step = video_fps / fps
        indices = np.arange(start_frame, end_frame, step).astype(int)
        
    if max_frames is not None and len(indices) > max_frames:
        indices = np.linspace(start_frame, end_frame, max_frames).astype(int)
        
    return indices

def process_one_frame(frame, max_resolution):
    """Resize and Base64 Encode"""
    if frame is None:
        return None
    
    original_height, original_width = frame.shape[:2]
    
    if max_resolution is not None and (original_width > max_resolution or original_height > max_resolution):
        n_width = int(max_resolution * original_width / max(original_width, original_height))
        n_height = int(max_resolution * original_height / max(original_width, original_height))
    else:
        n_width = original_width
        n_height = original_height
    
    # Resize
    resized = cv2.resize(frame, (n_width, n_height))
    # Encode
    _, buffer = cv2.imencode('.jpg', resized)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

def prepare_video_input(video_path, indices, max_resolution, max_workers=4):
    start_t = time.time()
    
    frame_queue = queue.Queue(maxsize=max_workers * 2)
    result_queue = queue.Queue()
    
    def producer():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        current_pos = indices[0]
        
        for idx in indices:
            if idx < current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                current_pos = idx
            
            while current_pos < idx:
                cap.grab()
                current_pos += 1
                
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps if fps > 0 else 0
                frame_queue.put((idx, frame, timestamp))
            current_pos += 1
        
        cap.release()
        # Signal workers to stop
        for _ in range(max_workers):
            frame_queue.put(None)

    def worker():
        while True:
            item = frame_queue.get()
            if item is None:
                frame_queue.task_done()
                break
            
            idx, frame, timestamp = item
            res = process_one_frame(frame, max_resolution)
            result_queue.put((idx, res, timestamp))
            frame_queue.task_done()

    # Start Producer
    prod_thread = threading.Thread(target=producer)
    prod_thread.start()
    
    # Start Workers
    workers = []
    for _ in range(max_workers):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)
        
    # Wait for producer to finish pushing
    prod_thread.join()
    # Wait for queue to be empty (all processed)
    frame_queue.join()
    
    # Collect results
    results_list = []
    while not result_queue.empty():
        results_list.append(result_queue.get())
    
    # Sort by index to ensure order
    results_list.sort(key=lambda x: x[0])
    
    frames = [x[1] for x in results_list]
    timestamps = [x[2] for x in results_list]
        
    cost = time.time() - start_t
    return frames, timestamps

def _ts_to_seconds(ts: str) -> float:
        """
        SRT 时间戳: 'HH:MM:SS,mmm' -> seconds(float)
        """
        # 00:01:02,345
        hh, mm, rest = ts.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def parse_srt(srt_path: str) -> str:
    """
    Parse SRT text directly into formatted string:
    start_time --> end_time
    text
    ...
    """
    if not os.path.exists(srt_path):
        return [], ''
    
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        srt_text = f.read()
    # 兼容不同换行
    srt_text = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    blocks = re.split(r"\n\s*\n", srt_text)
    sbt_items: List[Dict] = []

    time_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    )

    for block in blocks:
        lines = [ln.strip("\ufeff").strip() for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue

        # 常见格式:
        # 1
        # 00:00:01,000 --> 00:00:04,000
        # text...
        # 也可能没有序号行（少见），所以用正则查找时间行
        time_line_idx = None
        m = None
        for i, ln in enumerate(lines[:3]):  # 时间行一般在前几行
            m = time_re.search(ln)
            if m:
                time_line_idx = i
                break
        if time_line_idx is None or m is None:
            continue

        start_s = _ts_to_seconds(m.group("start"))
        end_s = _ts_to_seconds(m.group("end"))

        text_lines = lines[time_line_idx + 1 :]
        text = "\n".join(text_lines).strip()

        text = re.sub(r'<[^>]+>', '', text)
        # 可选：去掉少量常见的 HTML 实体（按需扩展）
        text = (
            text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        ).strip()

        if text:
            sbt_items.append({"start": start_s, "end": end_s, "text": text})

    # 确保按 start 排序
    sbt_items.sort(key=lambda x: (x["start"], x["end"]))
    
    # Convert to string format
    sbt_str = ''        
    for item in sbt_items:
        sbt_str += f'Start Time: {item["start"]:.1f} --> End Time: {item["end"]:.1f} Subtitle: {item["text"]}\n\n'
        
    return sbt_items, sbt_str


def get_subtitle_in_segment(subtitle_items: List[Dict], start_time, end_time):
    """
    Get subtitles in a segment.
    
    Args:
        subtitle_items: List of subtitle items
        start_time: Start time of the segment
        end_time: End time of the segment
        
    Returns:
        List of subtitle items in the segment
    """
    subtitles_in_segment = []
    for subtitle in subtitle_items:
        if start_time <= subtitle['start'] and end_time >= subtitle['end']:
            new_subtitle = {
                'start': subtitle['start'],
                'end': subtitle['end'],
                'offset': start_time,
                'shift_start': subtitle['start'] - start_time,
                'shift_end': subtitle['end'] - start_time,
                'text': subtitle['text']
            }
            subtitles_in_segment.append(new_subtitle)
    
    sbt_str = ''        
    for item in subtitles_in_segment:
        sbt_str += f'Start Time: {item["start"]:.1f} --> End Time: {item["end"]:.1f} Subtitle: {item["text"]}\n\n'
            
    return subtitles_in_segment, sbt_str


def get_video_property(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             raise Exception(f"Failed to open video {video_path}")
             
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"
        cap.release()
    except Exception as e:
        # logging.error(f"Failed to get metadata for {video_path}: {e}")
        fps = None
        duration = None
        resolution = None
    
    video_info = {
        'video_path': video_path,
        'duration': duration,
        'fps': fps,
        'resolution': resolution,
    }
    
    return video_info


def read_json(json_path: str) -> Dict:
    """
    Read JSON file and return as dictionary.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        json_data = json.load(f)
    return json_data