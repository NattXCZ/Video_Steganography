
import subprocess
import json
def get_frames_metadata(file):
    command = '"{ffexec}" -show_frames -print_format json "{filename}"'.format(ffexec='ffprobe', filename=file)
    response_json = subprocess.check_output(command, shell=True)#, stdeone)
    frames = json.loads(response_json)["frames"]
    frames_metadata, frames_type, frames_type_bool = [], [], []
    for frame in frames:
        if frame["media_type"] == "video":
            video_frame = json.dumps(dict(frame), indent=4)
            frames_metadata.append(video_frame)
            frames_type.append(frame["pict_type"])
            if frame["pict_type"] == "I":
                frames_type_bool.append(True)
            else:
                frames_type_bool.append(False)
    print(frames_type)
    return frames_metadata, frames_type, frames_type_bool

if __name__ == "__main__":
    video_path = 'input_video.mp4'
    frames_metadata, frames_type, frames_type_bool =    get_frames_metadata(video_path) 
    #print(frames_metadata)
    print(frames_type)
    #print(frames_type_bool)
    