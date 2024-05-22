from src.utils import w_binary as bnr
from src.utils import video_processing as vid_utils
from src.methods import hamming_code as hc
import numpy as np

def message_test():
    # text test
    file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    binary_string = bnr.file_to_binary_1D_arr(file_path)
    
    bnr.binary_1D_arr_to_file(binary_string, "./new_file.txt")
    
def message_length():

    xor_key = 3
    message = np.arange(1, 11)
    print(message)
    message1 = np.roll(message, xor_key)
    print(message1)
    

def test_append():
    file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\output\renewed\1ORIGin.png"
    binary_arr = bnr.file_to_binary_1D_arr(file_path)
    binary_array = np.append(binary_arr, [0, 0, 0])
    bnr.binary_1D_arr_to_file(binary_array, "./new_file.png")
    
    
    
def test_list():
    file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    binary_arr = bnr.file_to_binary_1D_arr(file_path)
    seznam = []
    for prvek in binary_arr:
        seznam.append(prvek)
        
    bnr.binary_1D_arr_to_file(np.array(seznam), "./new_file.txt")
        
        
        
def roll_test():
    message = np.array([1,2,3,4,5,6,7,8,9])
    tmp_message = np.roll(message, 3)
    print(tmp_message)
    message = np.roll(tmp_message, -3)
    
    print(message)
    
    
if __name__ == "__main__":
    #message_test()
    #message_length()
    
    #test_append()
    #test_list()
    #roll_test()
    file_path  =r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\frames\frame_1.png"
    binary_arr = bnr.file_to_binary_1D_arr(file_path)
    print(len(binary_arr))