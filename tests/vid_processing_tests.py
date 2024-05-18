from src.utils import video_processing as vid


def audio_test(f_path):
    print(vid.has_audio_track(f_path))
    
    
    
    
if __name__ == "__main__":
    video_file_path_horse = r"C:\Users\natal\OneDrive\Desktop\data__test\input\horse3s.mp4"
    video_file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\DSC_0588.MOV"
    
    audio_test(video_file_path)