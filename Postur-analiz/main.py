import cv2
import time
from pose_detection import preprocess_frame, run_inference, inference_fn
from exercise_detection import draw_keypoints_and_edges_press, draw_keypoints_and_edges_press_raise, draw_keypoints_for_hammer_curl

cap = cv2.VideoCapture(0)
time.sleep(2)
print("Lütfen Yapmak İstediğiniz Hareketi Seçiniz:\n 1. Dumbbell Overhead Press\n 2. Side Lateral Raise\n 3. Hummer Curl")
exercise_choice = int(input("Seçiminizi giriniz (1, 2 ya da 3): "))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        processed_frame = preprocess_frame(frame)
        keypoints_with_scores = run_inference(inference_fn, processed_frame)
        if exercise_choice == 1:
            draw_keypoints_and_edges_press(frame, keypoints_with_scores[0][0])
        elif exercise_choice == 2:
            draw_keypoints_and_edges_press_raise(frame, keypoints_with_scores[0][0])
        elif exercise_choice == 3:
            draw_keypoints_for_hammer_curl(frame, keypoints_with_scores[0][0])
        else:
            print("Geçersiz seçim, program sonlandırılıyor...")
            break

        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program sonlandırılıyor...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Kaynaklar serbest bırakıldı ve çıkış yapıldı.")