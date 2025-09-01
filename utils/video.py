import cv2
def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret,frame= cap.read()
        frames.append(frame)

    return frames

def save_video(output_frame,output_path):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'),
                          fps=24,
                          frame_size=(ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()