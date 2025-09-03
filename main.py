from utils import read_video, save_video
from tracking import Tracker
import cv2
import numpy as np
from team_to_player.team_to_player import TeamAssigner 
from ball_to_player import PlayerBallAssigner
from camera_move import CameraMovementEstimator
from view_transform import ViewTransformer
from speed_cal import SpeedAndDistance_Estimator
import time
import os
from collections import Counter

def get_video_path():
    """Get video path from user input"""
    while True:
        video_path = input("Enter video path: ").strip()
        video_path = video_path.strip('"').strip("'")
        
        if not video_path:
            print("Please enter a valid path")
            continue
            
        return video_path



def main():
    try:
        print("=== FOOTBALL TRACKING PROCESSING ===")
        start_time = time.time()
        
        video_path = get_video_path()
        

        print("1. Reading video...")
        video_frames = read_video(video_path)
        if not video_frames:
            print("Error: Cannot read video")
            return
        
        print("2. Initializing tracker...")
        tracker = Tracker('training/best_2.pt')
        
        print("3. Processing tracking...")
        tracks = tracker.get_object_tracks(video_frames,
                                           read_from_stub=False,
                                           stub_path='stubs/track_stubs.pkl')
        
        print("4. Adding positions to tracks...")
        tracker.add_position_to_tracks(tracks)

        print("5. Estimating camera movement...")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                  read_from_stub=False,
                                                                                  stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        print("6. Transforming view...")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        print("7. Interpolating ball positions...")
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        print("8. Calculating speed and distance...")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        print("9. Assigning player teams...")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                     track['bbox'],
                                                     player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        
        # Assign goalkeeper color
        for frame_num, frame_goalkeepers in enumerate(tracks['goalkeepers']):
            for goalkeeper_id, goalkeeper_data in frame_goalkeepers.items():
                tracks['goalkeepers'][frame_num][goalkeeper_id]['team_color'] = (255, 165, 0)

        # Assign referee color
        for frame_num, frame_referees in enumerate(tracks['referees']):
            for referee_id, referee_data in frame_referees.items():
                tracks['referees'][frame_num][referee_id]['team_color'] = (0, 255, 0)

        print("10. Assigning ball possession...")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox) if not np.isnan(ball_bbox[0]) else -1

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                    
        team_ball_control = np.array(team_ball_control, dtype=int)

        print("11. Drawing annotations...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        print("12. Drawing camera movement...")
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        print("13. Drawing speed and distance...")
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        print("14. Saving video...")
        os.makedirs('output_videos', exist_ok=True)
        
        input_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"output_videos/{input_filename}_processed.avi"
        
        save_video(output_video_frames, output_filename)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n=== COMPLETED! Total time: {total_time:.2f} seconds ===")
        print(f"Video saved at: {output_filename}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()