import cv2
import os

# Create a folder to save your training images
output_dir = "dataset_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# PATH TO YOUR VIDEO
video_path = "door_video2 .mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print("Instructions:")
print(" - Press 'SPACEBAR' to capture and save a frame.")
print(" - Press 'q' to quit.")

img_counter = 0

# Create a window that allows resizing
cv2.namedWindow("Extracting Frames from Video", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video file or failed to grab frame.")
        break
        
    # --- FIX THE ZOOM ISSUE ---
    # Resize the display frame to a standard width of 854 and height of 480 (480p)
    # This ensures it fits comfortably on any laptop or monitor screen
    display_frame = cv2.resize(frame, (854, 480))
    
    # Display the resized frame
    cv2.imshow("Extracting Frames from Video", display_frame)
    
    key = cv2.waitKey(30) & 0xFF
    
    # Press Spacebar to save the ORIGINAL high-quality frame
    if key == ord(' '):
        img_name = os.path.join(output_dir, f"frame_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)  # Saves original quality for AI training
        print(f"Saved original quality: {img_name}")
        img_counter += 1
        
    # Press 'q' to quit early
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()