import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.signal import wiener
from skimage.feature import hog
from skimage import exposure
import os

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing Application")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Store images
        self.images = {
            'original': None,
            'resized': None,
            'grayscale': None,
            'denoised': None,
            'enhanced': None,
            'sharpened': None,
            'deblurred': None,
            'hog_features': None,
            'final_result': None
        }
        
        # Parameters
        self.params = {
            'resize_dim': tk.IntVar(value=224),
            'gaussian_kernel': tk.IntVar(value=5),
            'gaussian_sigma': tk.DoubleVar(value=1.0),
            'unsharp_amount': tk.DoubleVar(value=1.5),
            'wiener_noise': tk.DoubleVar(value=0.01),
            'hog_orientations': tk.IntVar(value=9),
            'hog_pixels_per_cell': tk.IntVar(value=8),
            'hog_cells_per_block': tk.IntVar(value=2)
        }
        
        # Initialize GUI
        self.setup_gui()
        self.setup_menu()

    def setup_gui(self):
        # Control and display frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="ns")

        display_frame = ttk.Frame(self.root, padding=10)
        display_frame.grid(row=0, column=1, sticky="nsew")

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Set up controls and display
        self.setup_controls(control_frame)
        self.setup_display(display_frame)

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

    def setup_controls(self, parent):
        title = ttk.Label(parent, text="Image Processing Controls", style='Title.TLabel')
        title.pack(pady=10)

        # Scrollable frame for controls
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        upload_frame = ttk.LabelFrame(scrollable_frame, text="Image Input", padding="5")
        upload_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(upload_frame, text="Upload Image", command=self.upload_image).pack(fill=tk.X, pady=5)

        params_frame = ttk.LabelFrame(scrollable_frame, text="Processing Parameters", padding="5")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        param_entries = [
            ("Resize Dimension:", 'resize_dim'),
            ("Gaussian Kernel Size:", 'gaussian_kernel'),
            ("Gaussian Sigma:", 'gaussian_sigma'),
            ("Unsharp Amount:", 'unsharp_amount'),
            ("Wiener Noise Level:", 'wiener_noise'),
            ("HOG Orientations:", 'hog_orientations'),
            ("HOG Pixels per Cell:", 'hog_pixels_per_cell'),
            ("HOG Cells per Block:", 'hog_cells_per_block')
        ]
        
        for label_text, param_key in param_entries:
            frame = ttk.Frame(params_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label_text, width=25, anchor='w').pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=self.params[param_key]).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        ttk.Button(params_frame, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=10)
        ttk.Button(params_frame, text="Save All Images", command=self.save_all_images).pack(fill=tk.X, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_display(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.image_labels = {}
        labels = [
            ('Original', 'original'),
            ('Resized', 'resized'),
            ('Grayscale', 'grayscale'),
            ('Gaussian Smoothing', 'denoised'),
            ('Histogram Equalization', 'enhanced'),
            ('Unsharp Masking', 'sharpened'),
            ('Wiener Deblurred', 'deblurred'),
            ('HOG Features', 'hog_features'),
            ('Final Restored Image', 'final_result')
        ]
        
        # Arrange images in a grid (3 columns)
        columns = 3
        for idx, (label_text, key) in enumerate(labels):
            frame = ttk.LabelFrame(scrollable_frame, text=label_text, padding="5")
            row = idx // columns
            col = idx % columns
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            image_label = ttk.Label(frame)
            image_label.pack(padx=5, pady=5)
            self.image_labels[key] = image_label

            # Configure grid weights for equal resizing
            scrollable_frame.grid_columnconfigure(col, weight=1)
        
        # Allow rows to expand
        for row in range((len(labels) + columns - 1) // columns):
            scrollable_frame.grid_rowconfigure(row, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def display_image(self, key, image):
        # Convert the image to RGB if it's grayscale
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image for display if it's too large
        max_size = 300
        height, width = image_rgb.shape[:2]
        scaling_factor = min(max_size / width, max_size / height, 1)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image_rgb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
        
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_labels[key].imgtk = imgtk  # Keep a reference to avoid garbage collection
        self.image_labels[key].configure(image=imgtk)

    def upload_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
        )
        if filepath:
            image = cv2.imread(filepath)
            if image is None:
                messagebox.showerror("Error", "Failed to load the image. Please select a valid image file.")
                return
            self.images['original'] = image
            self.display_image('original', image)
            # Clear previous processed images
            for key in self.images:
                if key != 'original':
                    self.images[key] = None
                    self.image_labels[key].configure(image='')

    def resize_image(self):
        size = self.params['resize_dim'].get()
        if size <= 0:
            messagebox.showerror("Invalid Parameter", "Resize dimension must be positive.")
            return False
        self.images['resized'] = cv2.resize(self.images['original'], (size, size))
        self.display_image('resized', self.images['resized'])
        return True

    def convert_to_grayscale(self):
        if self.images['resized'] is None:
            messagebox.showerror("Error", "Resized image not available.")
            return False
        self.images['grayscale'] = cv2.cvtColor(self.images['resized'], cv2.COLOR_BGR2GRAY)
        self.display_image('grayscale', self.images['grayscale'])
        return True

    def apply_gaussian_smoothing(self):
        if self.images['grayscale'] is None:
            messagebox.showerror("Error", "Grayscale image not available.")
            return False
        kernel = self.params['gaussian_kernel'].get()
        if kernel % 2 == 0 or kernel <= 0:
            messagebox.showerror("Invalid Parameter", "Gaussian kernel size must be a positive odd integer.")
            return False
        sigma = self.params['gaussian_sigma'].get()
        if sigma <= 0:
            messagebox.showerror("Invalid Parameter", "Gaussian sigma must be positive.")
            return False
        self.images['denoised'] = cv2.GaussianBlur(self.images['grayscale'], (kernel, kernel), sigma)
        self.display_image('denoised', self.images['denoised'])
        return True

    def apply_histogram_equalization(self):
        if self.images['denoised'] is None:
            messagebox.showerror("Error", "Denoised image not available.")
            return False
        self.images['enhanced'] = cv2.equalizeHist(self.images['denoised'])
        self.display_image('enhanced', self.images['enhanced'])
        return True

    def apply_unsharp_masking(self):
        if self.images['enhanced'] is None:
            messagebox.showerror("Error", "Enhanced image not available.")
            return False
        kernel = self.params['gaussian_kernel'].get()
        amount = self.params['unsharp_amount'].get()
        if kernel % 2 == 0 or kernel <= 0:
            messagebox.showerror("Invalid Parameter", "Gaussian kernel size must be a positive odd integer.")
            return False
        if amount < 0:
            messagebox.showerror("Invalid Parameter", "Unsharp amount must be non-negative.")
            return False
        blurred = cv2.GaussianBlur(self.images['enhanced'], (kernel, kernel), 0)
        self.images['sharpened'] = cv2.addWeighted(self.images['enhanced'], 1 + amount, blurred, -amount, 0)
        self.display_image('sharpened', self.images['sharpened'])
        return True

    def apply_wiener_deblurring(self):
        if self.images['sharpened'] is None:
            messagebox.showerror("Error", "Sharpened image not available.")
            return False
        noise = self.params['wiener_noise'].get()
        if noise < 0:
            messagebox.showerror("Invalid Parameter", "Wiener noise level must be non-negative.")
            return False
        try:
            # Wiener filter expects float images
            image_float = self.images['sharpened'].astype(np.float64)
            deblurred = wiener(image_float, (5, 5), noise)
            deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
            self.images['deblurred'] = deblurred
            self.display_image('deblurred', self.images['deblurred'])
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Wiener deblurring failed: {e}")
            return False

    def extract_hog_features(self):
        if self.images['deblurred'] is None:
            messagebox.showerror("Error", "Deblurred image not available.")
            return False
        orientations = self.params['hog_orientations'].get()
        pixels_per_cell = self.params['hog_pixels_per_cell'].get()
        cells_per_block = self.params['hog_cells_per_block'].get()
        if orientations <= 0 or pixels_per_cell <= 0 or cells_per_block <= 0:
            messagebox.showerror("Invalid Parameter", "HOG parameters must be positive integers.")
            return False
        try:
            features, hog_image = hog(
                self.images['deblurred'],
                orientations=orientations,
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                visualize=True,
                block_norm='L2-Hys'
            )
            # Normalize hog_image to [0,255]
            hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255)).astype(np.uint8)
            self.images['hog_features'] = hog_image_rescaled
            self.display_image('hog_features', self.images['hog_features'])
            return True
        except Exception as e:
            messagebox.showerror("Error", f"HOG feature extraction failed: {e}")
            return False

    def create_final_result(self):
        if self.images['hog_features'] is None or self.images['deblurred'] is None:
            messagebox.showerror("Error", "Required images for final result are not available.")
            return False
        try:
            normalized_hog = cv2.normalize(self.images['hog_features'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Ensure both images are the same size
            if self.images['deblurred'].shape != normalized_hog.shape:
                normalized_hog = cv2.resize(normalized_hog, (self.images['deblurred'].shape[1], self.images['deblurred'].shape[0]))
            self.images['final_result'] = cv2.addWeighted(self.images['deblurred'], 0.7, normalized_hog, 0.3, 0)
            self.display_image('final_result', self.images['final_result'])
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Creating final result failed: {e}")
            return False

    def save_all_images(self):
        save_dir = filedialog.askdirectory()
        if save_dir:
            try:
                for key, image in self.images.items():
                    if image is not None:
                        # Convert grayscale images to BGR before saving to maintain consistency
                        if len(image.shape) == 2:
                            image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        else:
                            image_to_save = image
                        filepath = os.path.join(save_dir, f"{key}.png")
                        cv2.imwrite(filepath, image_to_save)
                messagebox.showinfo("Success", "Images saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save images: {e}")

    def process_image(self):
        if self.images['original'] is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        try:
            success = self.resize_image()
            if not success:
                return
            success = self.convert_to_grayscale()
            if not success:
                return
            success = self.apply_gaussian_smoothing()
            if not success:
                return
            success = self.apply_histogram_equalization()
            if not success:
                return
            success = self.apply_unsharp_masking()
            if not success:
                return
            success = self.apply_wiener_deblurring()
            if not success:
                return
            success = self.extract_hog_features()
            if not success:
                return
            success = self.create_final_result()
            if not success:
                return
            messagebox.showinfo("Success", "Image processing completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during processing: {e}")

def main():
    root = tk.Tk()
    
    # Optional: Customize styles
    style = ttk.Style()
    style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
    style.configure('TLabel', font=('Helvetica', 10))
    style.configure('TButton', font=('Helvetica', 10))
    style.configure('TLabelframe.Label', font=('Helvetica', 12, 'bold'))
    
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
