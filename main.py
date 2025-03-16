from utils import read_video, save_video
from trackers import Tracker
def main():
    videos_frames = read_video("input_videos/08fd33_4.mp4")
    

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(videos_frames, read_from_stub=True, 
                                       stub_path="stubs/track_stubs.pkl")

    # draw output
    output_frames = tracker.draw_annotations(videos_frames, tracks)
    
    save_video(output_frames, "output_videos/output.avi")



if __name__ == "__main__":
    main()