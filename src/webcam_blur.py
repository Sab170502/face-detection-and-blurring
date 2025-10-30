import cv2, os, urllib.request, numpy as np, sys, time

MODEL_DIR = "models"
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
CONF_THRESH = 0.5

PROTO_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
]

MODEL_URLS = [
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
]


def try_download(url, dest_path, desc=None, retries=2, timeout=20):
    desc = desc or os.path.basename(dest_path)
    print(f"Attempting to download {desc} from:\n  {url}")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Saved {desc} -> {dest_path}")
            return True
        except Exception as e:
            print(f"  Download attempt {attempt} failed: {e}")
            time.sleep(1)
    return False


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # prototxt
    if not os.path.exists(PROTO_PATH):
        for url in PROTO_URLS:
            if try_download(url, PROTO_PATH, desc="prototxt", retries=2):
                break
        else:
            print("\nCould not download prototxt automatically.")
            print(f"Please manually download the prototxt and save it as: {PROTO_PATH}")
            print("Manual download URL (open in browser):")
            print(
                "  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            )
            raise RuntimeError("prototxt missing")

    if not os.path.exists(MODEL_PATH):
        for url in MODEL_URLS:
            if try_download(url, MODEL_PATH, desc="caffemodel (~16MB)", retries=2):
                break
        else:
            print("\nAutomatic download of the caffemodel failed.")
            print("Please manually download and save as:", MODEL_PATH)
            print("Manual download URL (open in browser):")
            print(
                "  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            )
            raise RuntimeError("caffemodel missing")


def blur_region(img, ksize=(51, 51)):
    kx = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
    ky = ksize[1] if ksize[1] % 2 == 1 else ksize[1] + 1
    return cv2.GaussianBlur(img, (kx, ky), 0)


def main():
    try:
        ensure_models()
    except RuntimeError as e:
        print(
            "Model files are missing. Script will exit. See instructions above to download manually."
        )
        sys.exit(1)

    print("Loading network...")
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < CONF_THRESH:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            pad_w = int(0.12 * (x2 - x1))
            pad_h = int(0.12 * (y2 - y1))
            xa = max(0, x1 - pad_w)
            ya = max(0, y1 - pad_h)
            xb = min(w, x2 + pad_w)
            yb = min(h, y2 + pad_h)
            roi = frame[ya:yb, xa:xb]
            if roi.size == 0:
                continue
            frame[ya:yb, xa:xb] = blur_region(roi, ksize=(51, 51))
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (xa, max(12, ya - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        cv2.imshow("FaceShield (OpenCV DNN) - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
