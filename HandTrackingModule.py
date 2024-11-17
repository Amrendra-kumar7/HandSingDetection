import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=float(self.detectionCon),
                                        min_tracking_confidence=float(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self, lmList):
        if not lmList:
            return []
        fingers = []
        # Thumb: Compare x-coordinates because of its horizontal position
        fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

        # Other 4 fingers: Compare y-coordinates (tip vs. lower joint)
        for i in range(8, 21, 4):
            fingers.append(1 if lmList[i][2] < lmList[i - 2][2] else 0)
        return fingers


def detectDigit(fingers):
    digits = {
        (0, 0, 0, 0, 0): 0,
        (1, 0, 0, 0, 0): 1,
        (1, 1, 0, 0, 0): 2,
        (1, 1, 1, 0, 0): 3,
        (1, 1, 1, 1, 0): 4,
        (1, 1, 1, 1, 1): 5,
        (0, 0, 1, 1, 1): 6,  # Custom combination for 6
        (0, 1, 1, 1, 1): 7,
        (1, 0, 1, 1, 1): 8,
        (1, 1, 0, 1, 1): 9,
    }

    return digits.get(tuple(fingers), None)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Change camera index if needed
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            fingers = detector.fingersUp(lmList)
            digit = detectDigit(fingers)
            if digit is not None:
                print(f"Detected Digit: {digit}")
                # Display the detected digit
                cv2.putText(img, f"Digit: {digit}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q' key
            break


if __name__ == "__main__":
    main()

