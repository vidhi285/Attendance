import cv2

def test_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW helps on Windows
    if not cap.isOpened():
        print("Camera not opened for index", index)
        return False
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        return False
    print("Frame shape:", frame.shape)
    cap.release()
    return True

for idx in range(3):  # try 0,1,2
    ok = test_camera(idx)
    print(f"Index {idx} -> {ok}")
