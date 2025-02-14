import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# 禁用 PyAutoGUI 的暂停和失败安全机制
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 初始化 MediaPipe 绘图工具
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 定义控制区域
control_center_x, control_center_y = 640, 360
control_area_size = 600

# 初始化变量
prev_x, prev_y = pyautogui.position()
smoothing = 3
edge_acceleration = 2
velocity_x, velocity_y = 0, 0
last_click_time = 0
click_cooldown = 0.1
locked_position = None
is_dragging = False
last_scroll_time = 0
scroll_cooldown = 0.5
prev_middle_finger_y = None
scroll_state = 'idle'
bend_threshold = 10  # 弯曲阈值设为10度
scroll_threshold = 30  # 滚动阈值
pinch_threshold = 0.03  # 捏合阈值

# 新增：用于平滑的移动平均滤波器
smoothing_window = 10
x_buffer = deque(maxlen=smoothing_window)
y_buffer = deque(maxlen=smoothing_window)

def map_to_screen(x, y):
    in_min, in_max = 0, control_area_size
    out_min, out_max = 0, screen_width
    return int(np.interp(x, [in_min, in_max], [out_min, out_max])), \
           int(np.interp(y, [in_min, in_max], [out_min, out_max]))

def smooth_move(x, y, prev_x, prev_y):
    x_buffer.append(x)
    y_buffer.append(y)
    smooth_x = int(sum(x_buffer) / len(x_buffer))
    smooth_y = int(sum(y_buffer) / len(y_buffer))
    
    # 计算平滑后的坐标与前一个坐标的距离
    distance = np.sqrt((smooth_x - prev_x)**2 + (smooth_y - prev_y)**2)
    
    # 如果距离小于阈值，保持前一个坐标
    if distance < 5:  # 可以根据需要调整这个阈值
        return prev_x, prev_y
    
    return smooth_x, smooth_y

def accelerate_edge_movement(x, y, velocity_x, velocity_y):
    edge_zone = 50
    max_velocity = 100

    if x < edge_zone:
        velocity_x = min(velocity_x - edge_acceleration, -max_velocity)
    elif x > control_area_size - edge_zone:
        velocity_x = max(velocity_x + edge_acceleration, max_velocity)
    else:
        velocity_x = 0

    if y < edge_zone:
        velocity_y = min(velocity_y - edge_acceleration, -max_velocity)
    elif y > control_area_size - edge_zone:
        velocity_y = max(velocity_y + edge_acceleration, max_velocity)
    else:
        velocity_y = 0

    x += velocity_x
    y += velocity_y

    return x, y, velocity_x, velocity_y

def detect_finger_bend(hand_landmarks, finger_tip, finger_pip):
    tip = hand_landmarks.landmark[finger_tip]
    pip = hand_landmarks.landmark[finger_pip]
    mcp = hand_landmarks.landmark[finger_tip - 2]
    
    vec1 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
    vec2 = np.array([mcp.x - pip.x, mcp.y - pip.y, mcp.z - pip.z])
    
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.degrees(angle)

def detect_gestures(hand_landmarks):
    index_bend = detect_finger_bend(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_bend = detect_finger_bend(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)
    
    is_pinching = thumb_index_distance < pinch_threshold
    should_lock = index_bend > bend_threshold or middle_bend > bend_threshold
    
    return is_pinching, should_lock, middle_tip.y, middle_bend

try:
    print("增强手势鼠标控制已启动！")
    print("将手放在摄像头画面中心区域来控制鼠标")
    print("轻微弯曲食指或中指（超过10度）会锁定鼠标位置")
    print("捏合大拇指和食指进行点击和拖拽")
    print("快速上下翻动中指来滚动页面")
    print("按 ESC 键退出程序")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法获取摄像头画面。")
            continue

        # 水平翻转图像并转换为RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制控制区域
        cv2.rectangle(image, 
                      (control_center_x - control_area_size//2, control_center_y - control_area_size//2),
                      (control_center_x + control_area_size//2, control_center_y + control_area_size//2),
                      (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 获取食指指尖的坐标
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])

                # 检查食指是否在控制区域内
                if (control_center_x - control_area_size//2 < x < control_center_x + control_area_size//2 and
                    control_center_y - control_area_size//2 < y < control_center_y + control_area_size//2):
                    
                    # 转换坐标为相对于控制区域的坐标
                    x_rel = x - (control_center_x - control_area_size//2)
                    y_rel = y - (control_center_y - control_area_size//2)

                    # 加速边缘移动
                    x_rel, y_rel, velocity_x, velocity_y = accelerate_edge_movement(x_rel, y_rel, velocity_x, velocity_y)

                    # 将手指位置映射到屏幕坐标
                    screen_x, screen_y = map_to_screen(x_rel, y_rel)
                    
                    # 平滑鼠标移动
                    smooth_x, smooth_y = smooth_move(screen_x, screen_y, prev_x, prev_y)

                    # 检测手势
                    is_pinching, should_lock, middle_finger_y, middle_bend = detect_gestures(hand_landmarks)
                    
                    current_time = time.time()

                    # 处理锁定位置
                    if should_lock and not is_dragging:
                        if locked_position is None:
                            locked_position = (smooth_x, smooth_y)
                            print("锁定位置")
                    else:
                        if locked_position is not None:
                            print("解除锁定")
                        locked_position = None

                    # 处理点击和拖拽
                    if is_pinching:
                        if not is_dragging and current_time - last_click_time > click_cooldown:
                            if locked_position:
                                pyautogui.moveTo(*locked_position)
                            pyautogui.mouseDown()
                            is_dragging = True
                            last_click_time = current_time
                            print("开始拖拽")
                    elif is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                        print("结束拖拽")

                    # 处理滚动
                    if prev_middle_finger_y is not None:
                        if scroll_state == 'idle':
                            if abs(middle_finger_y - prev_middle_finger_y) > scroll_threshold / 1000:
                                scroll_state = 'scrolling'
                                scroll_amount = int((prev_middle_finger_y - middle_finger_y) * 1000)
                                pyautogui.scroll(scroll_amount)
                                print(f"滚动: {scroll_amount}")
                                last_scroll_time = current_time
                        elif scroll_state == 'scrolling':
                            if current_time - last_scroll_time > scroll_cooldown:
                                scroll_state = 'idle'

                    prev_middle_finger_y = middle_finger_y

                    # 移动鼠标
                    if not should_lock or is_dragging:
                        pyautogui.moveTo(smooth_x, smooth_y)
                    
                    # 更新前一帧的坐标
                    prev_x, prev_y = smooth_x, smooth_y

        # 显示图像
        cv2.imshow('Enhanced Gesture Mouse Control', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

except Exception as e:
    print(f"发生错误: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
