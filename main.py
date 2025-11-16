import cv2
import mediapipe as mp
import numpy as np
import imageio

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

monkey_img = cv2.imread('assets/monkey.jpg')

try:
    cat_gif = imageio.mimread('assets/cat_laughing.gif')
    cat_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in cat_gif]
except:
    cat_frames = None
    print("Could not load cat_laughing.gif")

try:
    dog_gif = imageio.mimread('assets/dog.gif')
    dog_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in dog_gif]
except:
    dog_frames = None
    print("Could not load dog.gif")

try:
    hello_gif = imageio.mimread('assets/hello.gif')
    hello_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in hello_gif]
except:
    hello_frames = None
    print("Could not load hello.gif")

try:
    nyancat_gif = imageio.mimread('assets/nyancat.gif')
    nyancat_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in nyancat_gif]
except:
    nyancat_frames = None
    print("Could not load nyancat.gif")


def is_peace_sign(landmarks):
    if len(landmarks) < 21:
        return False
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    return index_tip.y < ring_tip.y and index_tip.y < pinky_tip.y and middle_tip.y < ring_tip.y and middle_tip.y < pinky_tip.y


def is_index_finger_raised(landmarks):
    if len(landmarks) < 21:
        return False
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    return index_tip.y < middle_tip.y and index_tip.y < ring_tip.y and index_tip.y < pinky_tip.y and middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y


def is_fist(landmarks):
    if len(landmarks) < 21:
        return False
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    folded = sum(1 for tip, pip in zip(tip_ids, pip_ids) if landmarks[tip].y > landmarks[pip].y)
    return folded == 5


def is_palm(landmarks):
    if len(landmarks) < 21:
        return False
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    extended = sum(1 for tip, pip in zip(tip_ids, pip_ids) if landmarks[tip].y < landmarks[pip].y)
    return extended == 5


def is_pinky_raised(landmarks):
    if len(landmarks) < 21:
        return False
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]
    other_tips = [landmarks[i] for i in [4, 8, 12, 16]]
    return pinky_tip.y < pinky_mcp.y and all(pinky_tip.y < t.y for t in other_tips)


def is_ring_raised(landmarks):
    if len(landmarks) < 21:
        return False
    ring_tip = landmarks[16]
    ring_mcp = landmarks[13]
    other_tips = [landmarks[i] for i in [4, 8, 12, 20]]  # exclude ring
    return ring_tip.y < ring_mcp.y and all(ring_tip.y < t.y for t in other_tips)


def main():
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    gif_index = 0
    dog_index = 0
    hello_index = 0
    nyancat_index = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hand_results = hands.process(image_rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Index finger gesture
                if is_index_finger_raised(hand_landmarks.landmark) and monkey_img is not None:
                    h, w, _ = image.shape
                    xs = [lm.x * w for lm in hand_landmarks.landmark]
                    ys = [lm.y * h for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    hand_h = y_max - y_min
                    mh = min(hand_h * 2, monkey_img.shape[0])
                    mw = int(mh * monkey_img.shape[1] / monkey_img.shape[0])
                    resized = cv2.resize(monkey_img, (mw, mh))
                    sy = max(0, y_min - mh - 10)
                    ey = sy + mh
                    sx = x_min
                    ex = sx + mw
                    if ex <= w and ey <= h:
                        image[sy:ey, sx:ex] = resized

                # Fist gesture
                if is_fist(hand_landmarks.landmark) and cat_frames:
                    frame = cat_frames[gif_index]
                    gh = 200
                    gw = int(gh * frame.shape[1] / frame.shape[0])
                    resized = cv2.resize(frame, (gw, gh))
                    image[10:10+gh, 10:10+gw] = resized
                    gif_index = (gif_index + 1) % len(cat_frames)

                # Palm gesture
                if is_palm(hand_landmarks.landmark) and dog_frames:
                    frame = dog_frames[dog_index]
                    gh = 220
                    gw = int(gh * frame.shape[1] / frame.shape[0])
                    resized = cv2.resize(frame, (gw, gh))
                    image[230:230+gh, 10:10+gw] = resized
                    dog_index = (dog_index + 1) % len(dog_frames)

                # Pinky finger gesture
                if is_pinky_raised(hand_landmarks.landmark) and hello_frames:
                    frame = hello_frames[hello_index]
                    gh = 200
                    gw = int(gh * frame.shape[1] / frame.shape[0])
                    resized = cv2.resize(frame, (gw, gh))
                    image[10:10+gh, 250:250+gw] = resized
                    hello_index = (hello_index + 1) % len(hello_frames)

                # Ring finger gesture
                if is_ring_raised(hand_landmarks.landmark) and nyancat_frames:
                    frame = nyancat_frames[nyancat_index]
                    gh = 200
                    gw = int(gh * frame.shape[1] / frame.shape[0])
                    resized = cv2.resize(frame, (gw, gh))
                    image[250:250+gh, 10:10+gw] = resized
                    nyancat_index = (nyancat_index + 1) % len(nyancat_frames)

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
