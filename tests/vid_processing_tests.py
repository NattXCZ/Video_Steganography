from src.utils import video_processing as vid
import cv2
import numpy as np
import os

def audio_test(f_path):
    print(vid.has_audio_track(f_path))
    
    
def test_yuv(path_vid):
    vid.video_to_yuv_frames(path_vid)  
    
    
    
def read_one_pixel(path_vid, x, y):

    image = cv2.imread(path_vid)

    pixel_value = image[y, x]

    return pixel_value

def one_pixel_to_yuv(pic,x,y):
    image = cv2.imread(pic)
    YUV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(YUV_frame)
    
    print(f"Y = {Y[y, x]}")
    print(f"U = {U[y, x]}")
    print(f"V = {V[y, x]}")
    directory = os.path.dirname(pic)
    
    # Uložení Y, U, V komponent do souborů
    cv2.imwrite(os.path.join(directory, 'Y_component.png'), Y)
    cv2.imwrite(os.path.join(directory, 'U_component.png'), U)
    cv2.imwrite(os.path.join(directory, 'V_component.png'), V)

    
    
def change_img_value(path, x, y , new_val):
    image = cv2.imread(path)
    image[y, x][:] = new_val    # nastavi vsecky hodnoty v np.array na tu samou hodnotu
    cv2.imwrite(path, image)
    
def test_reading_YUV():
    
    #test_yuv(video_file_path)
    
    x = 1000
    y = 511
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\tmp\Y\frame_1.png"
    print(f"Y = {read_one_pixel(pic, x,y)}")
    
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\tmp\U\frame_1.png"
    print(f"U = {read_one_pixel(pic,  x,y)}")
    
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\tmp\V\frame_1.png"
    print(f"V = {read_one_pixel(pic,  x,y)}")
    
    
    y_folder = "./tmp/Y"
    u_folder = "./tmp/U"
    v_folder = "./tmp/V"
    output_folder = "./frames"
    


    
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\tmp\V\frame_1.png"
    print(f"V-orig = {read_one_pixel(pic,  x,y)}")
    change_img_value(pic,x,y, 210)
    print(f"V-changed = {read_one_pixel(pic,  x,y)}")
        
    vid.merge_yuv_to_rgb(y_folder, u_folder, v_folder, output_folder)
    
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\frames\frame_1.png"
    print(f"rgb = {read_one_pixel(pic,  x,y)}")
    
    #one_pixel_to_yuv(pic,x,y)
    
    
    
def keys_test():
    xor_key = np.array([1, 1, 1, 0, 0, 1, 1])
    key = 3
    array = np.array([0, 1, 0, 1, 0, 1, 1,1, 0, 1])
    print(f"orig = {array}")
    message = np.roll(array, key)
    print(f"msmsms = {message}")
        
        
        
        
def bin_testtt():

    picture_pathV = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\tmp\V\frame_1.png"
    imageV = cv2.imread(picture_pathV)
    row =1
    col = 1
    print(imageV[row, col])
    
if __name__ == "__main__":
    video_file_path_horse = r"C:\Users\natal\OneDrive\Desktop\data__test\input\horse3s.mp4"
    video_file_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\DSC_0588.MOV"
    
    #audio_test(video_file_path)
    
    #test_yuv(video_file_path)
    #test_reading_YUV()
    x = 1000
    y = 511
    pic = r"C:\Users\natal\gitHub_projecstNattX\Video_Steganography\frames\frame_1.png"
    #print(f"rgb = {read_one_pixel(pic,  x,y)}")
    
    #one_pixel_to_yuv(pic,x,y)

    #keys_test()
    bin_testtt()
