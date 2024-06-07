import numpy as np
import cv2
from scipy.fftpack import dct, idct

# Algorithm 1: Object Motion Detection
def object_motion_detection(i_frame, p_frame, b_frame):
    motion_vectors = []
    for frame in [p_frame, b_frame]:
        mv = np.sqrt((frame - i_frame) ** 2)
        motion_vectors.append(mv)
    motion_magnitude = np.sum(motion_vectors, axis=0)
    y_coords, x_coords = np.where(motion_magnitude <= 7)
    return x_coords, y_coords

# Algorithm 2: Setup of Threshold Values
def setup_threshold_values(D, T):
    thresholds = []
    for u in range(3):
        f = -T if D[2*u] < 0 else T
        s = -T if D[2*u+1] < 0 else T
        thresholds.append((f, s))
    return thresholds

# Convert text message to binary
def text_to_binary(message):
    return ''.join(format(ord(c), '08b') for c in message)

# Convert binary to text
def binary_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

# Algorithm 3: Hiding Technique
def hide_data(frame, x_coords, y_coords, f, s, message_bits):
    stego_frame = frame.copy()
    S = 0
    for x, y in zip(x_coords, y_coords):
        if S >= len(message_bits):
            break
        block = stego_frame[y:y+8, x:x+8]
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        D = block_dct.flatten()
        for u in range(3):
            if S < len(message_bits):
                if message_bits[S] == '1':
                    if abs(D[2*u]) < abs(D[2*u+1]):
                        D[2*u], D[2*u+1] = D[2*u+1] + s, D[2*u]
                    else:
                        D[2*u] += f
                else:
                    if abs(D[2*u]) < abs(D[2*u+1]):
                        D[2*u] += s
                    else:
                        D[2*u], D[2*u+1] = D[2*u+1], D[2*u] + f
                S += 1
        block_dct = D.reshape((8, 8))
        stego_frame[y:y+8, x:x+8] = idct(idct(block_dct.T, norm='ortho').T, norm='ortho')
    return stego_frame

# Algorithm 4: Extracting Technique
def extract_data(stego_frame, x_coords, y_coords, message_length):
    message_bits = []
    for x, y in zip(x_coords, y_coords):
        if len(message_bits) >= message_length:
            break
        block = stego_frame[y:y+8, x:x+8]
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        D = block_dct.flatten()
        for k in range(0, 6, 2):
            if len(message_bits) < message_length:
                if D[k] < D[k+1]:
                    message_bits.append('1')
                else:
                    message_bits.append('0')
    return ''.join(message_bits)

# Main program
if __name__ == "__main__":
    T = 20  # Example threshold value

    # Secret message
    secret_message = "Secret text message"
    message_bits = text_to_binary(secret_message)
    message_length = len(message_bits)

    # Open video file
    video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('stego_video.mp4', fourcc, fps, (width, height), isColor=False)

    # Read first frame (I-frame)
    ret, i_frame = cap.read()
    if not ret:
        print("Error reading I-frame.")
        exit()
    i_frame = cv2.cvtColor(i_frame, cv2.COLOR_BGR2GRAY)

    # Process each frame
    frame_count = 1
    while frame_count < total_frames - 1:
        # Read P-frame
        ret, p_frame = cap.read()
        if not ret:
            print("Error reading P-frame.")
            break
        p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

        # Read B-frame
        ret, b_frame = cap.read()
        if not ret:
            print("Error reading B-frame.")
            break
        b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)

        # Detect object motion
        x_coords, y_coords = object_motion_detection(i_frame, p_frame, b_frame)
        if len(x_coords) == 0 or len(y_coords) == 0:
            print("No motion detected in frame {}.".format(frame_count))
            continue

        # Setup threshold values
        D = np.random.randn(6)  # Example DCT coefficients
        thresholds = setup_threshold_values(D, T)

        # Hide data in P-frame
        stego_p_frame = hide_data(p_frame, x_coords, y_coords, thresholds[0][0], thresholds[0][1], message_bits)

        # Write stego P-frame to video
        out.write(cv2.cvtColor(stego_p_frame, cv2.COLOR_GRAY2BGR))

        # Update I-frame
        i_frame = p_frame.copy()
        frame_count += 2

    cap.release()
    out.release()

    # To extract data from stego video
    cap = cv2.VideoCapture('stego_video.mp4')
    ret, i_frame = cap.read()
    if not ret:
        print("Error reading stego I-frame.")
        exit()
    i_frame = cv2.cvtColor(i_frame, cv2.COLOR_BGR2GRAY)

    extracted_bits = []
    frame_count = 1
    while frame_count < total_frames - 1:
        # Read P-frame
        ret, p_frame = cap.read()
        if not ret:
            print("Error reading stego P-frame.")
            break
        p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

        # Read B-frame
        ret, b_frame = cap.read()
        if not ret:
            print("Error reading stego B-frame.")
            break
        b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)

        # Detect object motion
        x_coords, y_coords = object_motion_detection(i_frame, p_frame, b_frame)
        if len(x_coords) == 0 or len(y_coords) == 0:
            print("No motion detected in stego frame {}.".format(frame_count))
            continue

        # Extract data from P-frame
        extracted_bits.extend(extract_data(p_frame, x_coords, y_coords, message_length - len(extracted_bits)))

        # Update I-frame
        i_frame = p_frame.copy()
        frame_count += 2

    cap.release()

    extracted_message = binary_to_text(''.join(extracted_bits))
    
    print("Original Message:", secret_message)
    print("Extracted Message:", extracted_message)
