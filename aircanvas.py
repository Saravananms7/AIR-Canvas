import sys
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import math
import pytesseract
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time
import io
from PIL import Image
import os

# Set Tesseract path - update this path according to your installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AirCanvas(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Canvas with Handwriting Recognition")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for video and canvas
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create right panel for text recognition
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create labels for video and canvas
        self.video_label = QLabel()
        self.canvas_label = QLabel()
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.canvas_label)
        
        # Create text recognition widgets
        self.text_label = QLabel("Recognized Text:")
        self.text_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.recognized_text = QLabel()
        self.recognized_text.setStyleSheet("font-size: 12pt; background-color: white; padding: 10px; min-height: 200px;")
        self.recognized_text.setWordWrap(True)
        
        # Create control buttons
        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_text)
        self.recognize_button = QPushButton("Recognize Text")
        self.recognize_button.clicked.connect(self.process_text)
        
        # Add widgets to right panel
        right_layout.addWidget(self.text_label)
        right_layout.addWidget(self.recognized_text)
        right_layout.addWidget(self.recognize_button)
        right_layout.addWidget(self.clear_button)
        right_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize drawing variables
        self.bpoints = [deque(maxlen=1024)]
        self.gpoints = [deque(maxlen=1024)]
        self.rpoints = [deque(maxlen=1024)]
        self.ypoints = [deque(maxlen=1024)]
        
        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0
        
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.colorIndex = 0
        
        # Drawing state
        self.is_drawing = False
        self.was_drawing = False
        
        # Initialize canvas with correct data type (uint8)
        self.paintWindow = np.zeros((471,636,3), dtype=np.uint8) + 255
        self.paintWindow = cv2.rectangle(self.paintWindow, (40,1), (140,65), (0,0,0), 2)
        self.paintWindow = cv2.rectangle(self.paintWindow, (160,1), (255,65), self.colors[0], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (275,1), (370,65), self.colors[1], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (390,1), (485,65), self.colors[2], -1)
        self.paintWindow = cv2.rectangle(self.paintWindow, (505,1), (600,65), self.colors[3], -1)
        
        cv2.putText(self.paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = 33fps
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def clear_text(self):
        self.recognized_text.setText("")
        self.paintWindow[67:,:,:] = 255
        self.bpoints = [deque(maxlen=1024)]
        self.gpoints = [deque(maxlen=1024)]
        self.rpoints = [deque(maxlen=1024)]
        self.ypoints = [deque(maxlen=1024)]
        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0
    
    def process_text(self):
        try:
            # Create a copy of the canvas for processing, focusing on the drawing area
            canvas_copy = self.paintWindow[67:,:,:].copy()
            
            # Check if the canvas is empty (all white)
            if np.all(canvas_copy == 255):
                QMessageBox.information(self, "Info", "Canvas is empty. Please draw something first.")
                return
            
            # Convert to grayscale and apply noise reduction
            gray = cv2.cvtColor(canvas_copy, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Enhance contrast using CLAHE with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            gray = clahe.apply(gray)
            
            # Apply adaptive Gaussian thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Create kernels for morphological operations
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            
            # Apply advanced morphological operations
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
            thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
            
            # Find contours to identify text regions
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                QMessageBox.information(self, "Info", "No text regions detected. Try writing more clearly.")
                return
            
            # Create a mask for text regions with improved filtering
            mask = np.zeros_like(thresh)
            min_area = 100  # Increased minimum area threshold
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    # Filter based on aspect ratio to remove non-text shapes
                    if 0.1 < aspect_ratio < 15:
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Apply the mask and clean up
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
            
            # Save debug images
            debug_dir = 'debug_images'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            cv2.imwrite(os.path.join(debug_dir, 'original.png'), canvas_copy)
            cv2.imwrite(os.path.join(debug_dir, 'enhanced.png'), gray)
            cv2.imwrite(os.path.join(debug_dir, 'threshold.png'), thresh)
            
            # Scale up image for better OCR
            thresh = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # Add border to improve recognition
            thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            
            # Recognize text using Tesseract with enhanced settings for handwriting
            custom_config = r'--oem 3 --psm 6 -l eng --dpi 300 '\
                           r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? '\
                           r'-c tessedit_write_images=true '\
                           r'-c tessedit_enable_doc_dict=0 '\
                           r'-c textord_heavy_nr=1 '\
                           r'-c textord_min_linesize=2 '\
                           r'-c edges_max_children_per_outline=40 '\
                           r'-c tessedit_pageseg_mode=6 '\
                           r'-c tessedit_ocr_engine_mode=3 '\
                           r'-c load_system_dawg=0 '\
                           r'-c load_freq_dawg=0 '\
                           r'-c textord_force_make_prop_words=0 '\
                           r'-c tessedit_prefer_joined_punct=0 '\
                           r'-c tessedit_write_rep_codes=1 '\
                           r'-c tessedit_tess_adaption_mode=3 '\
                           r'-c tessedit_adaption_debug=1'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Process and display the recognized text
            if text.strip():
                current_text = self.recognized_text.text()
                if current_text:
                    self.recognized_text.setText(f"{current_text}\n{text.strip()}")
                else:
                    self.recognized_text.setText(text.strip())
            else:
                QMessageBox.information(self, "Info", "Text detection failed. Try writing larger and clearer.")

        except Exception as e:
            error_msg = str(e)
            print(f"Detailed error in text recognition: {error_msg}")
            if "TesseractNotFoundError" in error_msg:
                QMessageBox.critical(self, "Error", "Tesseract is not properly installed or configured.")
            else:
                QMessageBox.critical(self, "Error", "Failed to process text. Please try again.")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Add color buttons to frame
        frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
        frame = cv2.rectangle(frame, (160,1), (255,65), self.colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), self.colors[1], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), self.colors[2], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), self.colors[3], -1)
        
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

        center = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip and thumb coordinates
                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                h, w, c = frame.shape
                
                # Convert normalized coordinates to pixel coordinates
                index_finger_px = (int(index_finger.x * w), int(index_finger.y * h))
                thumb_px = (int(thumb.x * w), int(thumb.y * h))
                
                # Calculate distance between thumb and index finger
                distance = self.calculate_distance(index_finger_px, thumb_px)
                
                # Draw line between thumb and index finger
                cv2.line(frame, index_finger_px, thumb_px, (255, 0, 0), 2)
                
                # Draw circle at index finger tip
                cv2.circle(frame, index_finger_px, 5, (0, 255, 255), -1)
                
                # Update drawing state based on finger distance
                if distance < 50:  # Threshold for pinch gesture
                    self.is_drawing = False
                    cv2.putText(frame, "Drawing: OFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.is_drawing = True
                    cv2.putText(frame, "Drawing: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                center = index_finger_px
                
                # Check if finger is pointing at color selection buttons
                if center[1] <= 65:
                    if 40 <= center[0] <= 140:  # Clear Button
                        self.clear_text()
                    elif 160 <= center[0] <= 255:
                        self.colorIndex = 0  # Blue
                    elif 275 <= center[0] <= 370:
                        self.colorIndex = 1  # Green
                    elif 390 <= center[0] <= 485:
                        self.colorIndex = 2  # Red
                    elif 505 <= center[0] <= 600:
                        self.colorIndex = 3  # Yellow
                else:
                    if self.is_drawing:  # Only draw if drawing is enabled
                        # Create new stroke when starting to draw after pinching
                        if not self.was_drawing:
                            if self.colorIndex == 0:
                                self.bpoints.append(deque(maxlen=512))
                                self.blue_index += 1
                            elif self.colorIndex == 1:
                                self.gpoints.append(deque(maxlen=512))
                                self.green_index += 1
                            elif self.colorIndex == 2:
                                self.rpoints.append(deque(maxlen=512))
                                self.red_index += 1
                            elif self.colorIndex == 3:
                                self.ypoints.append(deque(maxlen=512))
                                self.yellow_index += 1
                        
                        # Add points to current stroke
                        if self.colorIndex == 0:
                            self.bpoints[self.blue_index].appendleft(center)
                        elif self.colorIndex == 1:
                            self.gpoints[self.green_index].appendleft(center)
                        elif self.colorIndex == 2:
                            self.rpoints[self.red_index].appendleft(center)
                        elif self.colorIndex == 3:
                            self.ypoints[self.yellow_index].appendleft(center)
        else:
            # Append the next deques when nothing is detected
            self.bpoints.append(deque(maxlen=512))
            self.blue_index += 1
            self.gpoints.append(deque(maxlen=512))
            self.green_index += 1
            self.rpoints.append(deque(maxlen=512))
            self.red_index += 1
            self.ypoints.append(deque(maxlen=512))
            self.yellow_index += 1
        
        # Update previous drawing state
        self.was_drawing = self.is_drawing

        # Draw lines of all the colors on the canvas and frame 
        points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], self.colors[i], 2)
                    cv2.line(self.paintWindow, points[i][j][k - 1], points[i][j][k], self.colors[i], 2)
        
        # Convert frames to QImage and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_frame = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_frame))
        
        # Convert canvas to RGB and display
        rgb_canvas = cv2.cvtColor(self.paintWindow, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_canvas.shape
        bytes_per_line = ch * w
        qt_canvas = QImage(rgb_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.canvas_label.setPixmap(QPixmap.fromImage(qt_canvas))
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AirCanvas()
    window.show()
    sys.exit(app.exec_())