
import cv2
import tkinter as tk
import threading
import torch
from PIL import Image, ImageTk
from torchvision import models

# AI MODEL
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()

# CAMERA
cap = cv2.VideoCapture(0)
frame = None
lock = threading.Lock()
busy = False
running = True

def camera():
    global frame
    while running:
        ok, img = cap.read()
        if ok:
            with lock:
                frame = img

def analyze():
    global busy

    with lock:
        img = None if frame is None else frame.copy()

    if img is None:
        root.after(0, lambda: result.config(text="No camera image"))
        root.after(0, lambda: set_busy(False))
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = preprocess(Image.fromarray(img)).unsqueeze(0)

    with torch.no_grad():
        scores = model(x).softmax(1)[0]
        values, indices = scores.topk(3)

    text = "\n".join(
        f"{weights.meta['categories'][i]}: {v.item():.1%}"
        for v, i in zip(values, indices)
    )

    root.after(0, lambda: result.config(text=text))
    root.after(0, lambda: set_busy(False))

def start_analysis():
    global busy
    if not busy:
        busy = True
        result.config(text="Analyzing...")
        threading.Thread(target=analyze, daemon=True).start()

def set_busy(value):
    global busy
    busy = value

def update():
    with lock:
        img = None if frame is None else frame.copy()

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((480, 360))
        photo = ImageTk.PhotoImage(img)
        video.config(image=photo)
        video.image = photo

    if running:
        root.after(30, update)

def close():
    global running
    running = False
    cap.release()
    root.destroy()

# GUI
root = tk.Tk()
root.title("AI Vision - PyTorch")

video = tk.Label(root)
video.pack()

result = tk.Label(root, text="Ready to recognize objects!", font=("Arial", 13))
result.pack(pady=10)

tk.Button(root, text="Analyze!", command=start_analysis).pack(pady=5)
tk.Button(root, text="Exit", command=close).pack(pady=5)

threading.Thread(target=camera, daemon=True).start()
root.after(30, update)
root.mainloop()
 