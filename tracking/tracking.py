from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        # Lấy ra danh sách các bbox, nếu không có thì dùng list chứa NaN
        ball_bboxes = []
        for x in ball_positions:
            bbox = x.get(1, {}).get('bbox', None)
            if bbox is None or len(bbox) == 0:
                ball_bboxes.append([np.nan, np.nan, np.nan, np.nan])
            else:
                ball_bboxes.append(bbox)
        
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Nội suy các giá trị bị thiếu
        # Sử dụng 'cubic' để có đường đi mượt mà, tự nhiên hơn cho quả bóng
        # Giới hạn chỉ nội suy cho các khoảng trống nhỏ (<= 10 frames)
        df_ball_positions = df_ball_positions.interpolate(method='cubic', limit_direction='both', limit=10)
        
        # Lấp các giá trị còn thiếu ở đầu (nếu có) bằng giá trị hợp lệ đầu tiên
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.2)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
            "goalkeepers": []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            name_to_cls_id = {v: k for k, v in class_names.items()}

            # Covert to supervision Detection format
            sv_detections = sv.Detections.from_ultralytics(detection)
            #
            # # Convert GoalKeeper to player object
            # for object_ind, class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            tracked_detections = self.tracker.update_with_detections(sv_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeepers"].append({})

            for tracked_obj in tracked_detections:
                bbox = tracked_obj[0].tolist()
                cls_id = tracked_obj[3]
                track_id = tracked_obj[4]

                if cls_id == name_to_cls_id.get('player'):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == name_to_cls_id.get('referee'):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == name_to_cls_id.get('goalkeeper'):
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            for detection_box in sv_detections:
                bbox = detection_box[0].tolist()
                cls_id = detection_box[3]

                if cls_id == name_to_cls_id.get('ball'):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def _validate_color(self, color):
        """
        Validate và convert màu sắc về định dạng OpenCV hợp lệ
        """
        try:
            # Nếu color là None hoặc empty, sử dụng màu mặc định
            if color is None:
                return (0, 255, 0)  # Xanh lá mặc định
            
            # Nếu là numpy array, convert về tuple
            if hasattr(color, 'tolist'):
                color = color.tolist()
            
            # Nếu là list hoặc tuple
            if isinstance(color, (list, tuple)):
                # Đảm bảo có đúng 3 giá trị
                if len(color) >= 3:
                    # Convert về int và clamp giá trị 0-255
                    # Input color đã là BGR, không cần hoán đổi
                    b = int(max(0, min(255, color[0])))
                    g = int(max(0, min(255, color[1]))) 
                    r = int(max(0, min(255, color[2])))
                    return (b, g, r)
                else:
                    return (0, 255, 0)  # Fallback
            
            # Nếu là số đơn (grayscale)
            if isinstance(color, (int, float)):
                val = int(max(0, min(255, color)))
                return (val, val, val)
            
            # Fallback cho các trường hợp khác
            return (0, 255, 0)
            
        except Exception as e:
            print(f"Warning: Color validation failed: {e}, using default green")
            return (0, 255, 0)

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Validate và convert màu sắc về định dạng đúng
        validated_color = self._validate_color(color)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=validated_color,
            thickness=4, # Tăng độ dày từ 2 lên 4
            lineType=cv2.LINE_4
        )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Validate màu sắc
        validated_color = self._validate_color(color)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, validated_color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        
        # Kiểm tra để tránh division by zero
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            # Nếu không có đội nào kiểm soát bóng, hiển thị 0%
            team_1_percentage = 0.0
            team_2_percentage = 0.0
        else:
            team_1_percentage = team_1_num_frames / total_frames
            team_2_percentage = team_2_num_frames / total_frames

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_percentage * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_percentage * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                # Lấy màu đội với fallback
                color = player.get("team_color", (0, 0, 255))
                
                # Debug: In thông tin màu nếu có vấn đề
                if frame_num == 0 and track_id <= 3:  # Chỉ debug vài player đầu
                    print(f"Player {track_id} color: {color}, type: {type(color)}")
                
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                color = referee.get("team_color", (0, 255, 255)) # Lấy màu từ track, fallback màu vàng
                frame = self.draw_ellipse(frame, referee["bbox"], color)

            # Draw Goalkeeper
            for track_id, goalkeeper in goalkeeper_dict.items():
                color = goalkeeper.get("team_color", (255, 0, 255)) # Lấy màu từ track, fallback màu hồng
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], color, track_id)

            # Draw ball
            for _, ball in ball_dict.items():
                ball_bbox = ball.get("bbox", [])
                # Bỏ qua việc vẽ bóng nếu tọa độ không hợp lệ (NaN)
                if ball_bbox and not np.isnan(ball_bbox[0]):
                    frame = self.draw_triangle(frame, ball_bbox, (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames