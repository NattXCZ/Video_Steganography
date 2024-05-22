import os
import magic
from src.methods import hamming_code
from src.utils import w_binary as bnr

import numpy as np

from src.methods import hamming_one_pic as hmp



def hamming_encode_test():
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    orig_video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\horse3s.mp4"
    output_path = f"./tmp/output.mp4"
    
    hamming_code.hamming_encode(orig_video_path, message_path,output_path)




def hamming_one_frame_test_decode():
    picture_path = r"C:\Users\natal\OneDrive\Desktop\data__test\output\renewed\1_original_for_testing.png"
    #message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    
    
    message_path =r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    orig_message_array = bnr.file_to_binary_1D_arr(message_path)

    output_path = r"./new_file.png"
    hmp.hamming_decode_one_frame(picture_path ,3,output_path)


def hamming_one_frame_test_encode():

    picture_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\frame_2.png"
    #message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    
    message_path =r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    output_path = r"./new_file.png"
    hmp.hamming_encode_one_frame(picture_path, message_path ,3)


if __name__ == "__main__":
    #hamming_one_frame_test_encode()
    
    hamming_one_frame_test_decode()