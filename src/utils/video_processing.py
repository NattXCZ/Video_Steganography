import os
import shutil
import cv2
from subprocess import run ,call ,STDOUT ,PIPE

def decode_fourcc(cc):
    """
    Decodes a FourCC (four-character code) into its string representation.

    Examples:
        >>> decode_fourcc(875967080.0)  # Numerical input (0x34363248)
        'h264'
        >>> decode_fourcc(1668703592.0)  # Numerical input (0x64656370)
        'hevc'
        >>> decode_fourcc(808596553.0)  # Numerical input (0x30323449)
        'I420'
    """
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    

def delete_tmp(path="./tmp"):
    """Delete the temporary folder and its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print("[INFO] tmp files are cleaned up")


def has_audio_track(filename, ffprobe_path = r".\src\utils\ffprobe.exe"):

    """Check if the given video file contains audio streams."""
    result = run([ffprobe_path, "-v", "error", "-show_entries",
                            "format=nb_streams", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=PIPE,
        stderr=STDOUT)
    return (int(result.stdout) -1)


def get_vid_stream_bitrate(filename, ffprobe_path = r".\src\utils\ffprobe.exe"):
    """Get the bitrate of the first video stream in bits per second."""
    result = run([ffprobe_path, "-v", "quiet", "-select_streams", "v:0", 
                            "-show_entries", "stream=bit_rate", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=PIPE,
        stderr=STDOUT)
    return int(result.stdout)


def extract_audio_track(video_file, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    call([ffmpeg_path, "-i", video_file, "-aq", "0", "-map", "a", "tmp/audio.wav"]) 
    print("[INFO] audio extracted")


def video_to_rgb_frames(video_path):
    """
    Extracts frames from a video file and saves them as individual image files into "tmp" folder.
    Save as .png files.

    Args:
        video_path (str): The path to the input video file.
        
    """

    if not os.path.exists("./tmp"):
        os.makedirs("tmp")
    temporal_folder="./tmp"
    print("[INFO] tmp directory is created")


    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Video file cannot be opened!")
        return

    video_properties = {
        "format": capture.get(cv2.CAP_PROP_FORMAT),
        "codec": capture.get(cv2.CAP_PROP_FOURCC),
        "container": capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        #f"frame_{frame_count:04d}.png"
        #! tady byla změna v názvu
        cv2.imwrite(os.path.join(temporal_folder, f"frame_{frame_count}.png"), frame)

    capture.release()

    print("[INFO] extraction finished")
    return video_properties


def video_to_yuv_frames(video_path):
    """
    Extracts frames from a video file and saves them as individual image files into "tmp"folder.

    Args:
        video_path (str): The path to the input video file.
        
    """
    #TODO: možna zde udelat extrakci zvuku?
    
    if not os.path.exists("./tmp"):
        os.makedirs("tmp")
    temporal_folder="./tmp"
    print("[INFO] tmp directory is created")


    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Video file cannot be opened!")
        return

    # Get video properties before processing frames
    video_properties = {
        "format": capture.get(cv2.CAP_PROP_FORMAT),
        "codec": capture.get(cv2.CAP_PROP_FOURCC),
        "container": capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    png_compression_level = 0 
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        #f"frame_{frame_count:04d}.png"
        #! tady byla změna v názvu
        frame_name = f"frame_{frame_count}.png"
      
        # Split frame into YUV components
        YUV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(YUV_frame)

        # Save Y, U, V components into respective folders
        y_path = os.path.join(temporal_folder, "Y")
        u_path = os.path.join(temporal_folder, "U")
        v_path = os.path.join(temporal_folder, "V")
        os.makedirs(y_path, exist_ok=True)
        os.makedirs(u_path, exist_ok=True)
        os.makedirs(v_path, exist_ok=True)

        cv2.imwrite(os.path.join(y_path, frame_name), Y,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(u_path, frame_name), U,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(v_path, frame_name), V,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])

    capture.release()

    print("[INFO] extraction finished")
    return video_properties

def reconstruct_video_from_rgb_frames(file_path, properties, ffmpeg_path = r".\src\utils\ffmpeg.exe"):

    #! file_path je tady jen skrz zvuk a koncovku
    fps = properties["fps"]
    codec =  decode_fourcc(properties["codec"])
    file_extension =  file_path.rsplit(".", 1)[1]
    bitrate = get_vid_stream_bitrate(file_path)

    if has_audio_track(file_path):
        #extract audio stream from video
        #call([ffmpeg_path, "-i", file_path, "-aq", "0", "-map", "a", "tmp/audio.wav"]) 
        extract_audio_track(file_path)
        
        #recreate video from frames (without audio)
        #call([ffmpeg_path, "-r", str(fps), "-i", "tmp/%d.png" , "-vcodec", str(codec), "-b", str(bitrate),"-crf", "0","-pix_fmt", "yuv420p", f"tmp/video.{file_extension}", "-y"])
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png" , "-vcodec", str(codec), "-b", str(bitrate),"-crf", "0","-pix_fmt", "yuv420p", f"tmp/video.{file_extension}", "-y"])

        #add audio to a recreated video
        call([ffmpeg_path, "-i", f"tmp/video.{file_extension}", "-i", "tmp/audio.wav","-q:v", "1", "-codec", "copy", f"video.{file_extension}", "-y"])
   
    else:
        #recreate video from frames (without audio)
        #call([ffmpeg_path, "-r", str(fps), "-i", "tmp/%d.png","-q:v", "1", "-vcodec", str(codec), "-b", str(bitrate), "-pix_fmt", "yuv420p", f"video.{file_extension}", "-y"])
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png","-q:v", "1", "-vcodec", str(codec), "-b", str(bitrate), "-pix_fmt", "yuv420p", f"video.{file_extension}", "-y"])

    print("[INFO] reconstruction is finished")
    

def merge_yuv_to_rgb(y_folder, u_folder, v_folder, output_folder):
    """
    Creates PNG frames by combining corresponding Y, U, and V frames.

    Args:
        y_folder (str): Path to the folder containing Y frames.
        u_folder (str): Path to the folder containing U frames.
        v_folder (str): Path to the folder containing V frames.
        output_folder (str): Path to the output folder where combined frames will be saved.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    y_files = sorted(os.listdir(y_folder))
    u_files = sorted(os.listdir(u_folder))
    v_files = sorted(os.listdir(v_folder))

    for y_file, u_file, v_file in zip(y_files, u_files, v_files):
        y_frame = cv2.imread(os.path.join(y_folder, y_file), cv2.IMREAD_GRAYSCALE)
        u_frame = cv2.imread(os.path.join(u_folder, u_file), cv2.IMREAD_GRAYSCALE)
        v_frame = cv2.imread(os.path.join(v_folder, v_file), cv2.IMREAD_GRAYSCALE)

        # merging Y, U, V frames into the one frame
        yuv_frame = cv2.merge([y_frame, u_frame, v_frame])

        # from YUV to RGB
        rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

        frame_name = os.path.splitext(y_file)[0] + ".png"
        cv2.imwrite(os.path.join(output_folder, frame_name), rgb_frame)
        
    print("[INFO] frames are merged")
        
def reconstruct_video_from_yuv_frames(file_path, properties, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    y_folder = "./tmp/Y"
    u_folder = "./tmp/U"
    v_folder = "./tmp/V"
    output_folder = "./frames"
    
    merge_yuv_to_rgb(y_folder, u_folder, v_folder, output_folder)
    
    reconstruct_video_from_rgb_frames(file_path, properties, ffmpeg_path)
    
    print("[INFO] video is reconstructed")