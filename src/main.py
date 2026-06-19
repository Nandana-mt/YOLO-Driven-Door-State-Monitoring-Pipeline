import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# Force RTSP to use TCP globally
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class OfficeDoorMonitor:
    """
    A unified, production-grade Computer Vision pipeline to track office entrance 
    states using an anti-flutter state machine on both RTSP feeds and local video files.
    """
    def __init__(self, source_type="rtsp", model_path=r"model\best.pt"):
        # =====================================================
        # ⚙️ SYSTEM CONFIGURATIONS
        # =====================================================
        self.source_type = source_type.lower()
        self.model_path = model_path
        
        # Hardcoded pipeline network/file parameters
        self.rtsp_url = "rtsp://iva:admin@123@192.168.123.10:554/Streaming/Channels/101"
        self.video_path = "door_video2.mp4"
        
        # Algorithmic thresholds
        self.confidence_threshold = 0.6
        self.required_frames = 5
        self.frame_skip = 5  # Only applied to local video files

        # =====================================================
        # 🔄 STATE MACHINE VARIABLES
        # =====================================================
        self.current_state = "Closed"
        self.last_opened_time = "Never"
        self.last_closed_time = "Never"  # New tracking point
        self.pending_state = None
        self.consecutive_count = 0
        self.frame_count = 0

        # Initialize the deep learning model backend
        self.model = YOLO(self.model_path)

    def _process_door_state(self, detected_classes):
        """
        Private helper: Evaluates frame-by-frame inference metrics 
        through the temporal anti-flutter state machine.
        """
        frame_state = None
        if "door open" in detected_classes:
            frame_state = "Open"
        elif "door close" in detected_classes:
            frame_state = "Closed"

        # --- ANTI-FLUTTER LOGIC ---
        if frame_state and frame_state != self.current_state:
            if frame_state == self.pending_state:
                self.consecutive_count += 1
            else:
                self.pending_state = frame_state
                self.consecutive_count = 1

            if self.consecutive_count >= self.required_frames:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Check which transition occurred to update the correct log
                if frame_state == "Open" and self.current_state == "Closed":
                    self.last_opened_time = timestamp
                elif frame_state == "Closed" and self.current_state == "Open":
                    self.last_closed_time = timestamp

                self.current_state = frame_state
                self.pending_state = None
                self.consecutive_count = 0

        elif frame_state == self.current_state:
            self.pending_state = None
            self.consecutive_count = 0

    def _open_source(self):
        """
        Private helper: Allocates the VideoCapture hardware descriptor handles.
        """
        if self.source_type == "rtsp":
            print(f"Connecting to RTSP stream:\n{self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            print(f"Opening video file:\n{self.video_path}")
            return cv2.VideoCapture(self.video_path)

    def _draw_ui(self, frame):
        """
        Private helper: Computes relative layout placement and draws dashboard graphics.
        """
        status_color = (0, 255, 0) if self.current_state == "Open" else (0, 0, 255)
        frame_height, frame_width = frame.shape[:2]

        # Box height increased to 150 to accommodate the extra metric line
        box_width, box_height = 460, 150
        box_x1 = frame_width - box_width - 20
        box_y1 = 20
        box_x2 = frame_width - 20
        box_y2 = box_y1 + box_height

        # Render background plate, status text, and timestamps
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        
        cv2.putText(frame, f"STATUS: {self.current_state.upper()}", (box_x1 + 20, box_y1 + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"LAST OPENED: {self.last_opened_time}", (box_x1 + 20, box_y1 + 85),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"LAST CLOSED: {self.last_closed_time}", (box_x1 + 20, box_y1 + 125),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return cv2.resize(frame, (854, 480))

    def run(self):
        """
        Starts the high-performance media loop orchestration thread.
        """
        cv2.namedWindow("Live Office Door Monitor", cv2.WINDOW_NORMAL)

        while True:
            cap = self._open_source()

            if not cap.isOpened():
                print("Failed to open video tracking target source pipeline.")
                if self.source_type == "video":
                    break  # Break out entirely if a local file track path is broken
                cv2.waitKey(5000)
                continue

            while cap.isOpened():
                # --- LIVE RTSP PIPELINE: DISCARD BACKLOG QUEUE LAGGARDS ---
                if self.source_type == "rtsp":
                    for _ in range(10):
                        if not cap.grab():
                            break

                ret, frame = cap.read()
                if not ret:
                    if self.source_type == "video":
                        print("End of video playback track file.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    print("Connection drop or packet frame loss detected. Reconnecting...")
                    break

                # --- CONTROL DECOUPLED FRAME-SKIPPING FLAGGING ---
                run_detection = True
                if self.source_type == "video":
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip != 0:
                        run_detection = False

                # --- MODEL PREDICTION EXECUTOR TRACK ---
                if run_detection:
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
                    detected_classes = [self.model.names[int(box.cls[0])] for box in results.boxes]
                    self._process_door_state(detected_classes)

                # --- COMPONENT CANVAS PRESENTER ---
                display_frame = self._draw_ui(frame)
                cv2.imshow("Live Office Door Monitor", display_frame)

                # Frame playback delay management
                wait_time = 20 if self.source_type == "video" else 1
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    print("Closing runtime framework allocation threads cleanly...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            
            if self.source_type == "video":
                break
                
            cv2.waitKey(5000)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = OfficeDoorMonitor(source_type="rtsp")
    monitor.run()