from utils import read_video, save_video
from trackers import Tracker
from team_assginer import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import cv2
import numpy as np
from camera_movement_estimator import CameraMovementEstimator





def main():
    videos_frames = read_video("input_videos/08fd33_4.mp4")
    

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(videos_frames, read_from_stub=True, 
                                       stub_path="stubs/track_stubs.pkl")

  
    #Estimate Camera Movement
    camera_movement_estimator = CameraMovementEstimator(videos_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(videos_frames,
                                                                               read_from_stub=True,
                                                             stub_path="stubs/camera_movement.pkl")
  
    #interploate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    #Assign Teams
    player_assigner = PlayerBallAssigner()
    teams_assigner = TeamAssigner()
    teams_assigner.assign_teams(videos_frames[0], tracks["players"][0])

    team_ball_control = []    
    for framen_num, player_track in enumerate(tracks["players"]):
       
        for player_id, track in player_track.items():
            team = teams_assigner.get_player_team(videos_frames[framen_num], track['bbox'], player_id)
            tracks['players'][framen_num][player_id]['team'] = team
            tracks['players'][framen_num][player_id]['team_color'] = teams_assigner.team_colors[team]
        #Assign Ball to Player
        ball_bbox = tracks["ball"][framen_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][framen_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][framen_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        

   
    
    output_frames = tracker.draw_annotations(videos_frames, tracks)

    #calculate team ball control
    team_ball_control = np.array(team_ball_control)
    team_ball_control_team1 = np.sum(team_ball_control == 1)
    team_ball_control_team2 = np.sum(team_ball_control == 2)
    team_ball_control1 = team_ball_control_team1/(team_ball_control_team1 + team_ball_control_team2)
    team_ball_control2 = team_ball_control_team2/(team_ball_control_team1 + team_ball_control_team2)


    #Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_frames, 
                                                                         camera_movement_per_frame)
    

    print("team ball control", team_ball_control1, team_ball_control2)
    save_video(output_video_frames, "output_videos/output.avi")



if __name__ == "__main__":
    main()