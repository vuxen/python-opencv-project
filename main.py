import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def index_up(landmarks):
    tip = landmarks[8].y
    pip = landmarks[6].y
    return tip < pip


def index_touch_mouth(landmarks, h, w):
    x = int(landmarks[8].x * w)
    y = int(landmarks[8].y * h)

    mouth_y_min = int(h * 0.40)
    mouth_y_max = int(h * 0.60)
    mouth_x_min = int(w * 0.40)
    mouth_x_max = int(w * 0.60)

    return mouth_x_min <= x <= mouth_x_max and mouth_y_min <= y <= mouth_y_max


def open_palm(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    extended = 0
    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            extended += 1

    return extended == 5


def index_point_chest_box(landmarks, h, w):
    x = int(landmarks[8].x * w)
    y = int(landmarks[8].y * h)

    chest_x_min = int(w * 0.35)
    chest_x_max = int(w * 0.65)
    chest_y_min = int(h * 0.60)
    chest_y_max = int(h * 0.85)

    return chest_x_min <= x <= chest_x_max and chest_y_min <= y <= chest_y_max, (chest_x_min, chest_y_min, chest_x_max, chest_y_max)


monkey_img = cv2.imread("assets/monkey.jpg")
monkeythink_img = cv2.imread("assets/monkeythinking.jpg")
chillman_img = cv2.imread("assets/chillman.jpg")
pointingme_img = cv2.imread("assets/pointingme.jpg")


def resize_to_match(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h))


def main():
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    current_display = np.zeros((300, 300, 3), dtype=np.uint8)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        chest_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                l = hand_landmarks.landmark

                chest_hit, chest_box = index_point_chest_box(l, h, w)

                if chest_hit:
                    current_display = pointingme_img
                    chest_detected = True
                elif open_palm(l):
                    current_display = chillman_img
                elif index_touch_mouth(l, h, w):
                    current_display = monkeythink_img
                elif index_up(l):
                    current_display = monkey_img

                x1, y1, x2, y2 = chest_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        resized_camera = resize_to_match(frame, 480)
        resized_display = resize_to_match(current_display, 480)

        combo = np.hstack((resized_camera, resized_display))

        cv2.imshow("Camera + Display", combo)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
