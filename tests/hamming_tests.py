import os
import magic
from src.methods import hamming_code
from src.utils import w_binary as bnr
from src.utils import video_processing as vid

import numpy as np

from src.methods import hamming_one_pic as hmp
from src.methods import hammin_vid as hmp_vid

from src.utils.bcolors import bcolors


def hamming_encode_test():
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    orig_video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\horse3s.mp4"
    output_path = f"./tmp/output.mp4"
    
    hamming_code.hamming_encode(orig_video_path, message_path,output_path)




def hamming_one_frame_test_decode():
    picture_path = r"C:\Users\natal\OneDrive\Desktop\data__test\output\renewed\1_original_for_testing.png"
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    
    
    #message_path =r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    orig_message_array = bnr.file_to_binary_1D_arr(message_path)

    output_path = r"./new_file.txt"
    hmp.hamming_decode_one_frame(picture_path ,3,output_path)


def hamming_one_frame_test_encode():

    picture_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\frame_2.png"
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    
    #message_path =r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    output_path = r"./new_file.txt"
    hmp.hamming_encode_one_frame(picture_path, message_path ,3)




def video_test_encode():
    #message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\DSC_0588_zaloha.MOV"

    
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    
    #message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\angel_secret.txt"
    
    properties = hmp_vid.hamming_encode(video_path, message_path , 3)
    
    
    output_path = r"./new_file.png"
    #apapapapap
    hmp_vid.hamming_decode_test(video_path, 3, properties, output_path)
    
    
def video_test_decode():
    video_path = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\video.MOV"

    output_path = r"./new_file.txt"
    hmp_vid.hamming_decode(video_path, 3, output_path)
    
    
    
    
def test_only_fisrt_frame_in_video_ENCODING():
    orig_video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\DSC_0588.MOV"
    
    vid_properties = vid.video_to_yuv_frames(orig_video_path)
    
    vid.reconstruct_video_from_yuv_frames(orig_video_path, vid_properties)
    
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    
    hmp.hamming_encode_one_frame("", message_path ,3)
    
    
def test_only_first_frame_in_video_DECODING():
    video_path = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\video.MOV"
    
    vid_properties = vid.video_to_yuv_frames(video_path)
    output_path = r"./new_file.txt"
    hmp.hamming_decode_one_frame("" ,3,output_path)
    
    
if __name__ == "__main__":
    #hamming_one_frame_test_encode()
    
    #hamming_one_frame_test_decode()
    
    
    video_test_encode()
    #video_test_decode()
    
    #test_only_fisrt_frame_in_video_ENCODING()
    #test_only_first_frame_in_video_DECODING()
    
    
    """codeword = np.array([1, 1, 1, 0, 0, 1, 1]) 
    print(np.array(bin(10)))
    codeword_xored = codeword ^ 10
    print(codeword_xored)
    print(codeword ^ 10 )"""
    
