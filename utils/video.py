import cv2
import tqdm

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {total_frames} frames, FPS: {fps}")
    
    frames = []
    print("Reading video...")
    
    with tqdm.tqdm(total=total_frames, desc="Đọc frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    
    cap.release()
    print(f"Read {len(frames)} frames")
    return frames

def save_video(output_frames, output_path):
    if not output_frames:
        print("No frames to save !!!")
        return
    
    #First frame size
    frame_height, frame_width = output_frames[0].shape[:2]
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'),
                          fps=24,
                          frameSize=(frame_width, frame_height))
    
    print(f"Saving video with size {frame_width}x{frame_height}...")
    with tqdm.tqdm(total=len(output_frames), desc="Lưu frames") as pbar:
        for frame in output_frames:
            out.write(frame)
            pbar.update(1)
    
    out.release()
    print(f"Saved video to: {output_path}")