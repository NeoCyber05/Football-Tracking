from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class TeamAssignerDBSCAN:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_player_color(self, frame, bbox):
        """
        Sử dụng DBSCAN để xác định màu áo chủ đạo của cầu thủ.
        DBSCAN giúp loại bỏ các pixel nhiễu và tìm ra cụm màu lớn nhất.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[int(image.shape[0] * 0.1):int(image.shape[0] / 2), :]
        
        # Reshape ảnh để clustering
        image_2d = top_half_image.reshape(-1, 3)
        if image_2d.shape[0] == 0:
            return np.array([0, 0, 0])

        # Sử dụng DBSCAN để tìm cụm màu chính
        # eps: khoảng cách tối đa giữa 2 điểm để được coi là lân cận.
        # min_samples: số lượng điểm tối thiểu trong lân cận để tạo thành một cụm.
        dbscan = DBSCAN(eps=20, min_samples=5, metric='euclidean')
        labels = dbscan.fit_predict(image_2d)

        # Tìm cụm lớn nhất (không tính các điểm nhiễu có label = -1)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        # Nếu không có cụm nào được tìm thấy (tất cả là nhiễu), trả về màu trung bình
        if len(counts) == 0:
            player_color = np.mean(image_2d, axis=0)
            return player_color

        # Lấy label của cụm có nhiều pixel nhất
        dominant_cluster_label = unique_labels[np.argmax(counts)]

        # Tính màu trung bình của cụm chủ đạo đó
        player_color = np.mean(image_2d[labels == dominant_cluster_label], axis=0)

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        # Lọc chỉ lấy players, loại bỏ referee và goalkeeper
        for _, player_detection in player_detections.items():
            class_name = player_detection.get("class_name", "player").lower()
            
            if class_name not in ["referee", "ref", "goalkeeper"]:
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
        
        if len(player_colors) >= 2:
            # Sử dụng KMeans để chia các màu áo của cầu thủ thành 2 đội
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(player_colors)

            self.kmeans = kmeans

            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
            
            # Thêm màu cho referee (vàng) và goalkeeper (hồng)
            self.team_colors["referee"] = [0, 255, 255]      # BGR: Yellow
            self.team_colors["goalkeeper"] = [255, 0, 255]  # BGR: Magenta

    def get_player_team(self, frame, player_bbox, player_id, class_name="player"):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        class_name_lower = class_name.lower()
        if class_name_lower in ["referee", "ref", "goalkeeper"]:
            self.player_team_dict[player_id] = class_name_lower
            return class_name_lower

        player_color = self.get_player_color(frame, player_bbox)

        # Dự đoán đội dựa trên mô hình KMeans đã huấn luyện ở `assign_team_color`
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id
        return team_id
