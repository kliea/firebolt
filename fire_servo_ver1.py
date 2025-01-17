import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import OrderedDict


class FireTracker:
    def __init__(self):
        self.fires = OrderedDict()  # Format: {fire_id: (timestamp, coordinates)}
        self.next_id = 0

    def update_fires(self, current_fires):
        current_time = time.time()
        # Remove old fires (more than 2 seconds without update)
        self.fires = OrderedDict(
            {k: v for k, v in self.fires.items() if current_time - v[0] < 2}
        )

        for fire_coords in current_fires:
            if not self.fires:  # If no existing fires, add as new
                self.fires[self.next_id] = (current_time, fire_coords)
                self.next_id += 1

        # Return coordinates of the earliest detected fire
        return self.fires[next(iter(self.fires))][1] if self.fires else None


class AngleCalculator:
    def __init__(self):
        # Camera specifications
        self.camera_width = 1920
        self.camera_height = 1080
        self.horizontal_fov = 62.2
        self.vertical_fov = 48.8

    def calculate_angles(self, x_pixel, y_pixel, frame_width, frame_height):
        if not (0 <= x_pixel < frame_width and 0 <= y_pixel < frame_height):
            return None, None

        # Convert pixel coordinates to normalized values (-1 to 1)
        x_norm = (x_pixel - frame_width / 2) / (frame_width / 2)
        y_norm = (y_pixel - frame_height / 2) / (frame_height / 2)

        # Calculate raw angles
        raw_pan = y_norm * (self.horizontal_fov / 2)
        raw_tilt = x_norm * (self.vertical_fov / 2)

        # Map pan angle from [-31.1, 31.1] to [0, 180]
        pan_angle = 90 + (raw_pan * 90 / (self.horizontal_fov / 2))

        # Map tilt angle considering the downward mounting
        tilt_angle = 90 - (raw_tilt * 90 / (self.vertical_fov / 2))

        # Clamp angles to valid servo range
        pan_angle = min(180, max(0, pan_angle))
        tilt_angle = min(180, max(0, tilt_angle))

        return pan_angle, tilt_angle


def initialize_model():
    # Load YOLOv8 model with optimization
    model = YOLO("YOLOv10-FireSmoke-X.pt")
    model.conf = 0.5  # Increase confidence threshold
    return model


def setup_camera():
    cap = cv2.VideoCapture(0)
    # Reduce frame rate
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap


def detect_fires(frame, model, processing_state):
    color = (0, 0, 255)
    """Handle only fire detection logic"""
    if time.time() - processing_state.get("last_process_time", 0) < 0.1:
        return None

    results = model(frame, verbose=False)
    current_fires = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                fire_center_x = (x1 + x2) / 2
                fire_center_y = (y1 + y2) / 2
                current_fires.append((fire_center_x, fire_center_y))
                break  # Only get the first fire detected

    processing_state["last_process_time"] = time.time()
    return current_fires[0] if current_fires else None


def main():
    # Initialize components
    model = initialize_model()
    cap = setup_camera()
    angle_calculator = AngleCalculator()
    pump_on = False
    processing_state = {"detecting": True, "last_detection_time": 0, "frame_count": 0}
    pan_angle, tilt_angle = 0.0, 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for better performance
            processing_state["frame_count"] += 1
            if processing_state["frame_count"] % 2 != 0:
                continue

            # Detect fire and calculate angles
            fire_coords = detect_fires(frame, model, processing_state)
            if fire_coords:
                pan_angle, tilt_angle = angle_calculator.calculate_angles(
                    fire_coords[0], fire_coords[1], frame.shape[1], frame.shape[0]
                )
                if pan_angle is not None and tilt_angle is not None and pump_on:

                    # call move servo
                    # set_servo_angles(int(pan_angle), int(tilt_angle))
                    time.sleep(2)
                    # turn off pump
                    pump_on = not pump_on
                elif pan_angle is not None and tilt_angle is not None and not pump_on:
                    pump_on = True
                    pass
                else:
                    pan_angle = None
                    tilt_angle = None
                    pump_on = False
            text = (
                f"Pan angle: {pan_angle}, tilt angle: {tilt_angle} pump on: {pump_on}"
            )
            print(
                f"Required angles - Pan: {pan_angle:.2f}°, Tilt: {tilt_angle:.2f}°, Pumping: {pump_on}"
            )
            cv2.putText(
                frame,
                text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # arduino.close()  # Close the serial connection


if __name__ == "__main__":
    main()

