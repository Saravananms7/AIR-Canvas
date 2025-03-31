import sys
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import math
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QMessageBox,
                           QInputDialog, QLineEdit, QFrame, QSizePolicy, QScrollArea,
                           QSplitter)
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
import io
from PIL import Image
import os
import requests

# Azure Computer Vision settings
AZURE_KEY = "F0vgrYFFvpP7G9128JTwarDncxzvJCGwpvcSJo7bZcg3aDx9EXiVJQQJ99BCACGhslBXJ3w3AAAFACOGLEVD"
AZURE_ENDPOINT = "https://aircanvas.cognitiveservices.azure.com/"

class OCRThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, canvas, vision_client):
        super().__init__()
        self.canvas = canvas
        self.vision_client = vision_client
        self.running = True
        
    def run(self):
        while self.running:
            if self.canvas is not None and self.vision_client is not None:
                try:
                    # Create a clean copy of the canvas without buttons
                    clean_canvas = self.canvas[70:,:,:].copy()
                    
                    # Check if there are any non-white pixels
                    gray = cv2.cvtColor(clean_canvas, cv2.COLOR_BGR2GRAY)
                    _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                    non_zero_pixels = cv2.countNonZero(thresholded)
                    
                    if non_zero_pixels < 20:  # Skip if canvas is empty
                        time.sleep(0.5)
                        continue
                        
                    # Enhance the drawing for OCR
                    enhanced = self.enhance_for_ocr(clean_canvas)
                    
                    # Save temporarily
                    debug_dir = 'debug_images'
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    image_path = os.path.join(debug_dir, 'realtime_ocr.png')
                    cv2.imwrite(image_path, enhanced)
                    
                    # Read and send to Azure
                    with open(image_path, 'rb') as image_file:
                        image_data = image_file.read()
                    
                    response = self.vision_client.read_in_stream(io.BytesIO(image_data), raw=True)
                    operation_url = response.headers["Operation-Location"]
                    
                    result = None
                    while self.running:
                        result_response = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY})
                        result_json = result_response.json()
                        
                        if result_json['status'] == 'succeeded':
                            result = result_json
                            break
                        elif result_json['status'] == 'failed':
                            break
                        time.sleep(0.5)
                    
                    if result and 'analyzeResult' in result:
                        extracted_text = "\n".join([line['text'] for line in result['analyzeResult']['readResults'][0]['lines']])
                        self.finished.emit(extracted_text)
                    
                except Exception as e:
                    print(f"OCR Error: {str(e)}")
            
            time.sleep(1)  # Process every second
    
    def enhance_for_ocr(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        enhanced = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        enhanced = 255 - enhanced
        return enhanced
    
    def stop(self):
        self.running = False
        self.wait()

class AirCanvas(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Canvas")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize Azure Computer Vision client
        self.vision_client = None
        self.initialize_azure_client()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create splitter to separate canvas and text panel
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Create canvas panel (left side)
        self.canvas_panel = QWidget()
        self.canvas_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout = QVBoxLayout(self.canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        
        # Create canvas label
        self.canvas_label = QLabel()
        self.canvas_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout.addWidget(self.canvas_label)
        
        # Create control buttons at bottom of canvas
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 10)
        
        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_canvas)
        self.clear_button.setStyleSheet("font-size: 14px; padding: 8px;")
        
        self.ai_text_button = QPushButton("AI Text")
        self.ai_text_button.clicked.connect(self.toggle_realtime_ocr)
        self.ai_text_button.setStyleSheet("font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;")
        
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.ai_text_button)
        canvas_layout.addWidget(control_panel)
        
        # Create text recognition panel (right side)
        self.text_panel = QFrame()
        self.text_panel.setFrameShape(QFrame.StyledPanel)
        self.text_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.text_panel.setMinimumWidth(400)
        text_layout = QVBoxLayout(self.text_panel)
        text_layout.setContentsMargins(10, 10, 10, 10)
        text_layout.setSpacing(10)
        
        # Text recognition widgets
        self.text_label = QLabel("Recognized Text:")
        self.text_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Create scroll area for recognized text
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.recognized_text = QLabel("Draw something and the recognized text will appear here...")
        self.recognized_text.setStyleSheet("""
            font-size: 14px; 
            background-color: white; 
            padding: 10px; 
            border: 1px solid #ddd;
            border-radius: 3px;
        """)
        self.recognized_text.setWordWrap(True)
        self.recognized_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll_area.setWidget(self.recognized_text)
        text_layout.addWidget(self.text_label)
        text_layout.addWidget(scroll_area)
        
        # Add stretch to push everything up
        text_layout.addStretch()
        
        # Add panels to splitter
        self.splitter.addWidget(self.canvas_panel)
        self.splitter.addWidget(self.text_panel)
        
        # Set initial splitter position (2/3 for canvas, 1/3 for text)
        self.splitter.setSizes([self.width()*2//3, self.width()//3])
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Drawing variables
        self.drawing_strokes = []
        self.current_stroke = None
        self.color = (0, 0, 255)  # Default color (red)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
        
        # Drawing state
        self.is_drawing = False
        self.last_point = None
        self.index_pos = None
        self.thumb_pos = None
        self.realtime_ocr_active = True  # Enabled by default now
        self.ocr_thread = None
        
        # Position smoothing buffers
        self.index_pos_buffer = deque(maxlen=5)  # Stores last 5 positions for smoothing
        self.thumb_pos_buffer = deque(maxlen=5)
        
        # Initialize canvas
        self.canvas = None
        self.reset_canvas()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        # Start OCR thread by default
        self.start_ocr_thread()

    def reset_canvas(self):
        """Reset the canvas to blank white"""
        width = self.canvas_label.width()
        height = self.canvas_label.height()
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.drawing_strokes = []
        self.current_stroke = None
        self.draw_color_buttons()

    def initialize_azure_client(self):
        try:
            if AZURE_KEY and AZURE_ENDPOINT:
                self.vision_client = ComputerVisionClient(
                    AZURE_ENDPOINT,
                    CognitiveServicesCredentials(AZURE_KEY)
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize Azure client: {str(e)}")

    def draw_color_buttons(self):
        """Draw color selection buttons at the top of the canvas"""
        if self.canvas is None:
            return
            
        # Clear the top area
        h, w = self.canvas.shape[:2]
        button_area_height = 70
        self.canvas[:button_area_height,:,:] = 255
        
        # Draw color selection buttons
        button_width = 100
        spacing = 20
        start_x = 20
        
        # Clear button
        cv2.rectangle(self.canvas, (start_x, 10), (start_x + button_width, 60), (0, 0, 0), 2)
        cv2.putText(self.canvas, "CLEAR", (start_x + 15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Color buttons
        for i, color in enumerate(self.colors):
            x1 = start_x + (i+1)*(button_width + spacing)
            x2 = x1 + button_width
            cv2.rectangle(self.canvas, (x1, 10), (x2, 60), color, -1)
            color_name = ["BLUE", "GREEN", "RED", "YELLOW"][i]
            text_color = (255, 255, 255) if i != 3 else (0, 0, 0)
            cv2.putText(self.canvas, color_name, (x1 + 15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

    def clear_canvas(self):
        """Clear the drawing area while keeping the buttons"""
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]
            self.canvas[70:h,:,:] = 255
        self.drawing_strokes = []
        self.current_stroke = None
        self.recognized_text.setText("Draw something and the recognized text will appear here...")

    def toggle_realtime_ocr(self):
        """Toggle real-time OCR processing"""
        self.realtime_ocr_active = not self.realtime_ocr_active
        
        if self.realtime_ocr_active:
            self.ai_text_button.setStyleSheet("font-size: 14px; padding: 8px; background-color: #f44336; color: white;")
            self.start_ocr_thread()
        else:
            self.ai_text_button.setStyleSheet("font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;")
            self.stop_ocr_thread()
        
        # Force UI update
        self.update()

    def start_ocr_thread(self):
        """Start the background OCR thread"""
        if self.vision_client is None:
            QMessageBox.warning(self, "Warning", "Azure client not initialized.")
            return
            
        if self.ocr_thread is None:
            self.ocr_thread = OCRThread(self.canvas, self.vision_client)
            self.ocr_thread.finished.connect(self.update_recognized_text)
            self.ocr_thread.start()

    def stop_ocr_thread(self):
        """Stop the background OCR thread"""
        if self.ocr_thread is not None:
            self.ocr_thread.stop()
            self.ocr_thread = None

    def update_recognized_text(self, text):
        """Update the recognized text in the UI"""
        if text.strip():  # Only update if we have non-empty text
            self.recognized_text.setText(text)

    def get_smoothed_position(self, buffer, new_pos):
        """Apply moving average smoothing to position coordinates"""
        if new_pos is None:
            return None
            
        buffer.append(new_pos)
        
        # Calculate average of all positions in buffer
        avg_x = sum(p[0] for p in buffer) / len(buffer)
        avg_y = sum(p[1] for p in buffer) / len(buffer)
        
        return (int(avg_x), int(avg_y))

    def draw_transparent_cursor(self, img, position, color, size=15, thickness=2):
        """Draw a transparent cursor at the given position"""
        if position is None or img is None:
            return
            
        # Create a temporary image for the cursor
        temp = img.copy()
        
        # Draw cursor (circle with crosshair)
        cv2.circle(temp, position, size//2, color, thickness)
        cv2.line(temp, 
                (position[0]-size, position[1]), 
                (position[0]+size, position[1]), 
                color, thickness)
        cv2.line(temp, 
                (position[0], position[1]-size), 
                (position[0], position[1]+size), 
                color, thickness)
        
        # Blend with original image
        alpha = 0.3
        cv2.addWeighted(temp, alpha, img, 1 - alpha, 0, img)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = self.hands.process(frame_rgb)
        
        # Reset raw positions
        raw_index_pos = None
        raw_thumb_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get finger coordinates
                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                h, w, _ = frame.shape
                
                # Convert to pixel coordinates
                raw_index_pos = (int(index_finger.x * w), int(index_finger.y * h))
                raw_thumb_pos = (int(thumb.x * w), int(thumb.y * h))
                
                # Apply smoothing to positions
                if raw_index_pos:
                    self.index_pos_buffer.append(raw_index_pos)
                    self.index_pos = self.get_smoothed_position(self.index_pos_buffer, raw_index_pos)
                
                if raw_thumb_pos:
                    self.thumb_pos_buffer.append(raw_thumb_pos)
                    self.thumb_pos = self.get_smoothed_position(self.thumb_pos_buffer, raw_thumb_pos)
                
                # Calculate distance between fingers
                if self.index_pos and self.thumb_pos:
                    distance = math.sqrt((self.index_pos[0]-self.thumb_pos[0])**2 + 
                                       (self.index_pos[1]-self.thumb_pos[1])**2)
                    
                    # Update drawing state
                    self.is_drawing = distance > 50  # Threshold for drawing
                    
                    # Handle color selection buttons
                    if self.index_pos[1] <= 70 and self.canvas is not None:
                        button_width = 100
                        spacing = 20
                        start_x = 20
                        
                        # Clear button
                        if start_x <= self.index_pos[0] <= start_x + button_width:
                            self.clear_canvas()
                        # Color buttons
                        for i in range(4):
                            x1 = start_x + (i+1)*(button_width + spacing)
                            x2 = x1 + button_width
                            if x1 <= self.index_pos[0] <= x2:
                                self.color = self.colors[i]
                    
                    # Handle drawing
                    if self.is_drawing and self.index_pos[1] > 70 and self.canvas is not None:
                        if self.last_point is None:
                            self.last_point = self.index_pos
                            self.current_stroke = {
                                'color': self.color,
                                'points': [self.index_pos]
                            }
                            self.drawing_strokes.append(self.current_stroke)
                        elif self.current_stroke is not None:
                            # Draw line from last point to current point
                            cv2.line(self.canvas, self.last_point, self.index_pos, self.color, 5)
                            self.current_stroke['points'].append(self.index_pos)
                            self.last_point = self.index_pos
                    else:
                        self.last_point = None
                        self.current_stroke = None
        
        # Display the canvas with cursors
        if self.canvas is not None:
            display_canvas = self.canvas.copy()
            
            # Draw transparent cursors (only if we have valid positions)
            if self.index_pos:
                self.draw_transparent_cursor(display_canvas, self.index_pos, (0, 0, 255))  # Red for index
            if self.thumb_pos:
                self.draw_transparent_cursor(display_canvas, self.thumb_pos, (0, 255, 0))  # Green for thumb
            
            # Convert to QImage and display
            h, w, ch = display_canvas.shape
            bytes_per_line = ch * w
            qt_image = QImage(display_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.canvas_label.setPixmap(QPixmap.fromImage(qt_image))

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        self.reset_canvas()

    def closeEvent(self, event):
        self.stop_ocr_thread()
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AirCanvas()
    window.show()
    sys.exit(app.exec_())