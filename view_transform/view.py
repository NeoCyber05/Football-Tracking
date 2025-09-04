import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32
        # Source points from the camera view (in pixels)
        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        # Destination points for the 2D top-down view
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate the perspective transformation matrix
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1])) # Transforms a single point from camera view to 2D view
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Apply the perspective transform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    # Add a new key with the transformed position
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed