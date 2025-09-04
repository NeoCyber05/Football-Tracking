import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    Manages team assignments for players.
    It first learns the two main team colors from an initial frame,
    then assigns a team to each player based on their jersey color.
    """
    def __init__(self):
        self.team_colors = {}
        self.player_team_map = {}
        self.kmeans = None
    
    def get_player_color(self, frame, bbox):
        """
        Gets the main color of a player's jersey.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Use the top half of the image to focus on the jersey
        top_half_image = image[0:int(image.shape[0]/2), :]

        if top_half_image.size == 0:
            return np.array([0, 0, 0])

        # Reshape the image for KMeans
        image_2d = top_half_image.reshape(-1, 3)
        
        # Find the two main colors in the player's image
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(image_2d)
        
        # The two colors found
        local_color_1 = kmeans.cluster_centers_[0]
        local_color_2 = kmeans.cluster_centers_[1]

        # If team colors are not learned yet, guess the player color
        if not self.team_colors:
            labels = kmeans.labels_
            clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
            # Assume the background is the most common color in the image corners
            corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
            player_cluster = 1 - non_player_cluster
            return kmeans.cluster_centers_[player_cluster]

        # Compare the two local colors with the known team colors
        dist_1_to_team1 = np.linalg.norm(local_color_1 - self.team_colors[1])
        dist_1_to_team2 = np.linalg.norm(local_color_1 - self.team_colors[2])
        min_dist_1 = min(dist_1_to_team1, dist_1_to_team2)

        dist_2_to_team1 = np.linalg.norm(local_color_2 - self.team_colors[1])
        dist_2_to_team2 = np.linalg.norm(local_color_2 - self.team_colors[2])
        min_dist_2 = min(dist_2_to_team1, dist_2_to_team2)

        # The color that is closer to a team color is the player's jersey color
        player_color = local_color_1 if min_dist_1 < min_dist_2 else local_color_2
        
        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Learns the two team colors by looking at all players in the first frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)
        
        # Save the main KMeans model
        self.kmeans = kmeans
        
        # Save the two team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assigns a team to a single player.
        """
        # If we already know the team for this player, return it
        if player_id in self.player_team_map:
            return self.player_team_map[player_id]
        
        # Get the player's jersey color
        player_color = self.get_player_color(frame, player_bbox)
        
        # Predict the team using the main KMeans model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        
        # Save the result for the next time we see this player
        self.player_team_map[player_id] = team_id
        
        return team_id