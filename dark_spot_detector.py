import cv2
import numpy as np
import os

# --- Configuration ---

# Path to the Haar Cascade XML file
# IMPORTANT: Adjust this path if you place the file elsewhere
HAARCASCADE_PATH = 'cascade_files/haarcascade_frontalface_default.xml'

# Parameters for dark spot detection
# Fine-tune these values based on your camera and lighting
# HSV format: H(0-179), S(0-255), V(0-255)
# Focus on low V (Value/Brightness) to find dark pixels
DARK_SPOT_COLOR_LOWER = np.array([0, 0, 0])
DARK_SPOT_COLOR_UPPER = np.array([179, 255, 45]) # V < 60 isolates dark areas
MIN_SPOT_AREA = 50  # Minimum contour area (in pixels) to consider as a spot

# --- Core Functions ---

def detect_dark_spots(face_image):
    """
    Analyzes the detected face region for dark spots.
    
    Args:
        face_image (np.array): The cropped image containing only the face.
        
    Returns:
        tuple: (modified_face_image, spot_count)
    """
    
    # 1. Color Space Conversion and Dark Spot Isolation
    hsv_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for dark pixels
    dark_spot_mask = cv2.inRange(hsv_face, DARK_SPOT_COLOR_LOWER, DARK_SPOT_COLOR_UPPER)
    
    # 2. Noise Cleaning (Morphology)
    kernel = np.ones((3, 3), np.uint8)
    
    # Opening: Remove small white noise from the mask
    cleaned_mask = cv2.morphologyEx(dark_spot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: Fill small gaps inside the spots
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Detect Contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spot_count = 0
    
    # 4. Display/Mark Spots
    for contour in contours:
        # Filter by area to remove very small specks/noise
        if cv2.contourArea(contour) > MIN_SPOT_AREA:
            # Get the minimum enclosing circle to mark the spot
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw a circle around the detected spot (e.g., in a magenta color)
            cv2.circle(face_image, center, radius, (255, 0, 255), 2)
            
            spot_count += 1
            
    return face_image, spot_count

def main():
    # --- Setup and Checks ---
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"ERROR: Haar Cascade file not found at '{HAARCASCADE_PATH}'")
        print("Please download 'haarcascade_frontalface_default.xml' and place it in the 'cascade_files/' directory.")
        return

    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    
    # Initialize the video capture object (0 is usually the default camera)
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("ERROR: Could not open video stream or file.")
        return

    print("--- Face Dark Spot Detector Initialized ---")
    print("Press 'q' to exit the live camera feed.")

    # --- Live Loop ---
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Flip the frame horizontally for a typical mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Face Detection
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )

        total_spots = 0

        # 2. Process Detected Faces
        for (x, y, w, h) in faces:
            # Crop the face region (Region of Interest - ROI)
            face_roi = frame[y:y+h, x:x+w]
            
            # Draw a green bounding box around the whole face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Apply the dark spot detection algorithm
            processed_face, count = detect_dark_spots(face_roi)
            total_spots += count
            
            # Replace the face ROI in the main frame with the processed face
            frame[y:y+h, x:x+w] = processed_face
            
        
        # 3. Display Results
        
        # Display the spot count on the frame
        cv2.putText(frame, f'Spots Detected: {total_spots}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Live Face Dark Spots Detector', frame)
        
        # Exit condition: Press 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ensure the cascade directory exists
    if not os.path.exists('cascade_files'):
        os.makedirs('cascade_files')
    
    main()