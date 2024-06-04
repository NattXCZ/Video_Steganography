import os

import numpy as np
import cv2

from src.methods import hamming_code
from src.utils import w_binary as bnr


if __name__ == "__main__":
    message_file = r"angel_secret.txt"
    message_text = "Lorem ipsum dolor sit amet congue. Ut consectetuer lorem est labore dolor. Ut quis assum erat at dolores zzril justo aliquyam aliquyam nibh eirmod labore est voluptua te sit illum. Feugiat eirmod vel no invidunt facilisis. Sed amet et vero consetetur eos sea tempor luptatum voluptua. Dolore labore sanctus et justo erat sed rebum accusam lorem illum elitr accusam takimata nonumy. Sanctus erat consequat nobis nisl amet eleifend et rebum justo lorem clita nisl. Sit velit eleifend sadipscing rebum ex. Veniam et lorem aliquyam sanctus exerci consetetur. No clita amet eirmod vulputate duo vero amet sadipscing. Magna nobis stet nonumy et aliquam. Gubergren accusam ipsum facilisi sed."
    video_path = r"cute_kitten_playing_w_tail.mov"
   
    key_1 = 10
    key_2 = 20
    key_3 = 2

    output_path = r"./decoded_message.txt"
    
    
    #ridi jestli vkladame nebo extrahujeme
    flag = 2
    
    
    if flag == 1:
        #zakodovani souboru do videa
        hamming_code.hamming_encode(video_path,message_file,key_1,key_2,key_3)
        
    elif flag == 2:

        #zakodovani retezce do videa
        hamming_code.hamming_encode(video_path,message_file,key_1,key_2,key_3,True)

    elif flag == 3:
        #dekodovani souboru z videa
        stego_video_path = r"video.avi"
        message_len_file = len(bnr.file_to_binary_1D_arr(message_file))
        hamming_code.hamming_decode(stego_video_path, key_1,key_2,key_3,message_len_file, output_path)

    elif flag == 4:
        #dekodovani retezce z videa
        stego_video_path = r"video.avi"
        message_len_string = len(bnr.string_to_binary_array(message_text))
        
        hamming_code.hamming_decode(stego_video_path, key_1,key_2,key_3,message_len_string, output_path, True)



    
    
    

