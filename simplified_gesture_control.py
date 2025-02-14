import cv2
import mediapipe as mp
import subprocess
import time
import webbrowser

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 配置
url = "https://chat.openai.com/"
website_opened = False
last_detection_time = 0
detection_interval = 0.5  # 保持0.5秒的检测间隔以提高响应速度

# 定义手势状态
GESTURE_UNKNOWN = "unknown"
GESTURE_PALM = "palm"
GESTURE_FIST = "fist"
previous_gesture = GESTURE_UNKNOWN

# 检测手掌打开的手势
def detect_open_palm(hand_landmarks):
    if hand_landmarks:
        fingertips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        fingers_extended = all(fingertip.y < wrist.y for fingertip in fingertips)
        palm_facing_camera = abs(palm_center.z) < 0.1
        
        return fingers_extended and palm_facing_camera
    return False

# 检测拳头手势
def detect_fist(hand_landmarks):
    if hand_landmarks:
        fingertips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                     hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
        
        middle_joints = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]]
        
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        fingers_curled = all(fingertip.y > middle_joint.y for fingertip, middle_joint in zip(fingertips, middle_joints))
        palm_facing_camera = abs(palm_center.z) < 0.1
        
        return fingers_curled and palm_facing_camera
    return False

# 获取当前手势
def get_current_gesture(hand_landmarks):
    if detect_open_palm(hand_landmarks):
        return GESTURE_PALM
    elif detect_fist(hand_landmarks):
        return GESTURE_FIST
    return GESTURE_UNKNOWN

# 打开网页函数
def open_website(url):
    global website_opened
    try:
        print(f"尝试打开网页: {url}")
        webbrowser.open(url)
        website_opened = True
        print(f"网页已打开: {url}")
    except Exception as e:
        print(f"打开网页时发生错误: {e}")

# 关闭网页函数
def close_website(url):
    global website_opened
    try:
        print(f"尝试关闭网页: {url}")
        applescript = f'''
        tell application "Google Chrome"
            set windowList to every window
            repeat with aWindow in windowList
                set tabList to every tab of aWindow
                repeat with atab in tabList
                    if URL of atab starts with "{url}" then
                        set index of aWindow to 1
                        set active tab index of aWindow to (get index of atab)
                        close active tab of aWindow
                        return
                    end if
                end repeat
            end repeat
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript])
        website_opened = False
        print(f"网页已关闭: {url}")
    except Exception as e:
        print(f"关闭网页时发生错误: {e}")

try:
    print("手势控制已启动！")
    print("从手掌打开变为其他手势 -> 打开 ChatGPT")
    print("从拳头变为其他手势 -> 关闭 ChatGPT")
    print("按 ESC 键退出程序")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法获取摄像头画面。")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_gesture = GESTURE_UNKNOWN
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = get_current_gesture(hand_landmarks)

        cv2.putText(image, f"Gesture: {current_gesture}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            if previous_gesture != GESTURE_UNKNOWN and current_gesture != previous_gesture:
                if previous_gesture == GESTURE_PALM and not website_opened:
                    print("检测到手掌打开手势变化，正在打开 ChatGPT...")
                    open_website(url)
                elif previous_gesture == GESTURE_FIST and website_opened:
                    print("检测到拳头手势变化，正在关闭 ChatGPT...")
                    close_website(url)
                last_detection_time = current_time

        previous_gesture = current_gesture

        cv2.imshow('Gesture Control', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

except Exception as e:
    print(f"发生错误: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
