# blur_utils.py
import cv2

def blur_region(img, ksize=(51,51)):
    # ksize must be odd numbers
    kx, ky = (ksize[0] | 1, ksize[1] | 1)
    return cv2.GaussianBlur(img, (kx, ky), 0)

def pixelate_region(img, blocks=10):
    h, w = img.shape[:2]
    x_steps = max(1, w // blocks)
    y_steps = max(1, h // blocks)
    temp = cv2.resize(img, (x_steps, y_steps), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
