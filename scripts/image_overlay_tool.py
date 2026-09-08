import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Button, Label, Frame, Listbox, END
from PIL import Image, ImageTk
import os

class ImageOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像叠加工具")
        
        self.img1 = None
        self.img2 = None
        self.overlay_img = None
        self.canvas = None
        self.photo = None
        
        self.alpha = 0.5
        self.bev_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        
        self.satellite_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_images"
        self.bev_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/streetview_images/semantic_pointcloud"
        
        self.create_widgets()
    
    def load_images_list(self):
        self.satellite_images = sorted([f for f in os.listdir(self.satellite_dir) if f.endswith('.png')])
        self.bev_images = sorted([f for f in os.listdir(self.bev_dir) if f.endswith('.png') and 'bev' in f.lower()])
    
    def create_widgets(self):
        self.load_images_list()
        
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        Label(left_frame, text="卫星图像", font=('Arial', 10, 'bold')).pack(pady=5)
        self.sat_listbox = Listbox(left_frame, width=35, height=10)
        self.sat_listbox.pack()
        for img in self.satellite_images:
            self.sat_listbox.insert(END, img)
        self.sat_listbox.bind('<<ListboxSelect>>', self.on_sat_select)
        
        Label(left_frame, text="BEV图像", font=('Arial', 10, 'bold')).pack(pady=5)
        self.bev_listbox = Listbox(left_frame, width=35, height=10)
        self.bev_listbox.pack()
        for img in self.bev_images:
            self.bev_listbox.insert(END, img)
        self.bev_listbox.bind('<<ListboxSelect>>', self.on_bev_select)
        
        right_frame = Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        control_frame = Frame(right_frame)
        control_frame.pack(pady=5, fill=tk.X)
        
        self.alpha_scale = Scale(control_frame, from_=0, to=100, orient='horizontal', 
                                label='透明度 (%)', command=self.update_alpha, length=200)
        self.alpha_scale.set(50)
        self.alpha_scale.pack(pady=2)
        
        self.scale_scale = Scale(control_frame, from_=10, to=200, orient='horizontal', 
                                label='BEV尺寸 (%)', command=self.update_bev_scale, length=200)
        self.scale_scale.set(100)
        self.scale_scale.pack(pady=2)
        
        btn_frame = Frame(control_frame)
        btn_frame.pack(pady=5)
        
        btn_reset = Button(btn_frame, text="重置位置", command=self.reset_position, width=10)
        btn_reset.grid(row=0, column=0, padx=3)
        
        btn_center = Button(btn_frame, text="居中", command=self.center_bev, width=10)
        btn_center.grid(row=0, column=1, padx=3)
        
        btn_save = Button(btn_frame, text="保存结果", command=self.save_result, width=10)
        btn_save.grid(row=0, column=2, padx=3)
        
        self.status_label = Label(control_frame, text="请从左侧列表选择图像")
        self.status_label.pack(pady=5)
        
        self.canvas = tk.Canvas(right_frame, width=600, height=500, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
    
    def on_sat_select(self, event):
        selection = self.sat_listbox.curselection()
        if selection:
            filename = self.satellite_images[selection[0]]
            path = os.path.join(self.satellite_dir, filename)
            self.img1 = cv2.imread(path)
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
            self.center_bev()
    
    def on_bev_select(self, event):
        selection = self.bev_listbox.curselection()
        if selection:
            filename = self.bev_images[selection[0]]
            path = os.path.join(self.bev_dir, filename)
            self.img2 = cv2.imread(path)
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
            self.center_bev()
    
    def update_alpha(self, value):
        self.alpha = int(value) / 100.0
        self.update_overlay()
    
    def update_bev_scale(self, value):
        self.bev_scale = int(value) / 100.0
        self.center_bev()
    
    def center_bev(self):
        if self.img1 is None or self.img2 is None:
            return
        
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        
        canvas_w, canvas_h = 600, 500
        scale = min(canvas_w / w1, canvas_h / h1)
        display_w, display_h = int(w1 * scale), int(h1 * scale)
        
        bev_scaled_w = int(w2 * scale * self.bev_scale)
        bev_scaled_h = int(h2 * scale * self.bev_scale)
        
        self.offset_x = (display_w - bev_scaled_w) // 2
        self.offset_y = (display_h - bev_scaled_h) // 2
        
        self.update_overlay()
    
    def update_overlay(self):
        if self.img1 is None or self.img2 is None:
            self.status_label.config(text="请选择两张图像")
            return
        
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        
        canvas_w, canvas_h = 600, 500
        scale = min(canvas_w / w1, canvas_h / h1)
        display_w, display_h = int(w1 * scale), int(h1 * scale)
        
        img1_resized = cv2.resize(self.img1, (display_w, display_h))
        
        bev_scaled_w = int(w2 * scale * self.bev_scale)
        bev_scaled_h = int(h2 * scale * self.bev_scale)
        img2_resized = cv2.resize(self.img2, (bev_scaled_w, bev_scaled_h))
        
        result = img1_resized.copy()
        
        y1 = max(0, self.offset_y)
        y2 = min(display_h, self.offset_y + bev_scaled_h)
        x1 = max(0, self.offset_x)
        x2 = min(display_w, self.offset_x + bev_scaled_w)
        
        overlay_y1 = max(0, -self.offset_y)
        overlay_y2 = overlay_y1 + (y2 - y1)
        overlay_x1 = max(0, -self.offset_x)
        overlay_x2 = overlay_x1 + (x2 - x1)
        
        if y1 < y2 and x1 < x2:
            alpha = self.alpha
            result[y1:y2, x1:x2] = (
                alpha * img2_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2] +
                (1 - alpha) * result[y1:y2, x1:x2]
            ).astype(np.uint8)
        
        self.overlay_img = result
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(result))
        self.canvas.delete("all")
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.status_label.config(
            text=f"透明度: {int(self.alpha*100)}% | BEV尺寸: {int(self.bev_scale*100)}% | 偏移: ({self.offset_x}, {self.offset_y})"
        )
    
    def on_press(self, event):
        self.dragging = True
        self.last_x = event.x
        self.last_y = event.y
    
    def on_drag(self, event):
        if self.dragging and self.img2 is not None:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_x = event.x
            self.last_y = event.y
            self.update_overlay()
    
    def on_release(self, event):
        self.dragging = False
    
    def reset_position(self):
        self.offset_x = 0
        self.offset_y = 0
        self.update_overlay()
    
    def save_result(self):
        if self.overlay_img is not None:
            path = os.path.join(self.satellite_dir, "overlay_result.png")
            cv2.imwrite(path, cv2.cvtColor(self.overlay_img, cv2.COLOR_RGB2BGR))
            self.status_label.config(text=f"已保存: {path}")

def main():
    root = tk.Tk()
    root.geometry("900x600")
    app = ImageOverlayApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()