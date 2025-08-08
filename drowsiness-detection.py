import cv2
from ultralytics import YOLO

model = YOLO("drowsiness_detection.pt")

def detect_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        annotated = results[0].plot()
        cv2.imshow("Drowsiness Detection", annotated)
        if cv2.waitKey(1) == 27:  
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_image(image_path, save_output=False, output_path="output.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image file: {image_path}")
        return
    results = model.predict(image)
    annotated = results[0].plot()
    cv2.imshow("Drowsiness Detection", annotated)
    if save_output:
        cv2.imwrite(output_path, annotated)
        print(f"Saved output image to {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(video_path, save_output=False, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if save_output:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        annotated = results[0].plot()

        if save_output:
            out.write(annotated)

        cv2.imshow("Drowsiness Detection", annotated)
        if cv2.waitKey(1) == 27:  
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def main():
    print("Select mode:")
    print("1 - Webcam")
    print("2 - Image")
    print("3 - Video")
    mode = input("Enter choice (1/2/3): ").strip()

    if mode == "1":
        detect_webcam()
    elif mode == "2":
        path = input("Enter image path: ")
        save = input("Save output? (y/n): ").lower() == "y"
        output_path = "output.jpg"
        if save:
            op = input("Enter output file path (default: output.jpg): ").strip()
            if op:
                output_path = op
        detect_image(path, save, output_path)
    elif mode == "3":
        path = input("Enter video path: ")
        save = input("Save output? (y/n): ").lower() == "y"
        output_path = "output.mp4"
        if save:
            op = input("Enter output file path (default: output.mp4): ").strip()
            if op:
                output_path = op
        detect_video(path, save, output_path)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
