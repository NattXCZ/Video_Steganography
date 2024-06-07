import numpy as np
import cv2
import random


float_pairs = ((3.60, 0.65), (2.92, 0.95), (3.26, 1.38), (0.73, 3.73), 
               (2.46, 3.72), (2.33, 1.98), (1.46, 2.06), (1.09, 0.63))

float_pairs = ((3.60, 0.65), (3.60, 0.65), (3.60, 0.65), (3.60, 0.65),
                (3.60, 0.65), (3.60, 0.65), (3.60, 0.65), (3.60, 0.65))

def logistic_key(N, mu, x0):
    X = np.zeros(N)
    X[0] = mu * x0 * (1 - x0) 
    #X[0] =round(abs(mu * x0 * (1 - x0)), 4)
    for k in range(1, N):
        X[k] = mu * X[k-1] * (1 - X[k-1])
    return X


def process_part_components(orig_img, new_img, start_row, start_col, h_step, w_step, B):
    k = 0
    for i in range(start_row, start_row + h_step):
        for j in range(start_col, start_col + w_step):
            C = orig_img[i, j]
            B_val = 255 if B[k] == 1 else B[k]
            new_img[i, j] = C  ^ B_val
            
            k += 1
            
    return new_img
            
            
def create_B(logistic_seq):
    T = np.mean(logistic_seq)
    B = np.array([1 if x >= T else 0 for x in logistic_seq], dtype=np.uint8)
    return B


def process_image(img_path, output_path):
    #processing image in grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    height, width = image.shape
    if height % 4 != 0 or width % 2 != 0:
        raise ValueError("Image dimensions must be divisible by 4 for height and 2 for width")
    
    #image to 8 parts
    h_step = height // 4
    w_step = width // 2
    
    
    # Initializing the encrypted image
    encryptedImage = np.zeros_like(image, dtype=np.uint8)

    i = 0
    for float_pair in float_pairs:
        generatedKey = logistic_key((height * width) // 8, float_pair[0], float_pair[1])
        start_row = (i // 2) * h_step
        start_col = (i % 2) * w_step
        process_part_components(image, encryptedImage, start_row, start_col, h_step, w_step, create_B(generatedKey))
        i += 1
        

    png_compression_level = 0
    cv2.imwrite(output_path, encryptedImage, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
    
    cv2.imshow('Encrypted Image', encryptedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    flag = 1

    if flag == 1:
        path = r"carousel-horse.png"
        output_path = r"chaos_image.png"
        process_image(path, output_path)
    
    elif flag ==2:
        
        path = r"chaos_image.png"
        output_path = r"chaos_image2.png"
        process_image(path, output_path)
    
    elif flag == 3:
        
        img_path = r"carousel-horse.png"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        output_path1 = r"carousel_horse_gray.png"

        png_compression_level = 0
        cv2.imwrite(output_path1, image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        
    
