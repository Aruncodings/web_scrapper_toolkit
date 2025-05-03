# gui.py - GUI for the Article Summarizer
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import threading
import logging
import requests
import sys
from main import ArticleProcessor, UserDB

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class AuthWindow:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = ctk.CTkFrame(root, width=400, height=300)
        self.frame.pack(pady=50, padx=50, fill="both", expand=True)

        self.username = ctk.StringVar()
        self.password = ctk.StringVar()
        self.email = ctk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Login Frame
        self.login_frame = ctk.CTkFrame(self.frame)
        self.login_frame.pack(pady=20, padx=40, fill="both", expand=True)

        ctk.CTkLabel(self.login_frame, text="Username:").pack(pady=(20, 0))
        self.username_entry = ctk.CTkEntry(self.login_frame, textvariable=self.username)
        self.username_entry.pack()

        ctk.CTkLabel(self.login_frame, text="Password:").pack(pady=(10, 0))
        self.password_entry = ctk.CTkEntry(self.login_frame, textvariable=self.password, show="*")
        self.password_entry.pack()

        ctk.CTkButton(self.login_frame, text="Login", command=self.login).pack(pady=20)
        ctk.CTkButton(self.login_frame, text="Register", command=self.show_register).pack()

    def show_register(self):
        self.login_frame.pack_forget()
        self.create_register_frame()

    def create_register_frame(self):
        self.register_frame = ctk.CTkFrame(self.frame)
        self.register_frame.pack(pady=20, padx=40, fill="both", expand=True)

        ctk.CTkLabel(self.register_frame, text="Username:").pack(pady=(20, 0))
        ctk.CTkEntry(self.register_frame, textvariable=self.username).pack()

        ctk.CTkLabel(self.register_frame, text="Password:").pack(pady=(10, 0))
        ctk.CTkEntry(self.register_frame, textvariable=self.password, show="*").pack()

        ctk.CTkLabel(self.register_frame, text="Email:").pack(pady=(10, 0))
        ctk.CTkEntry(self.register_frame, textvariable=self.email).pack()

        ctk.CTkButton(self.register_frame, text="Register", command=self.register).pack(pady=20)
        ctk.CTkButton(self.register_frame, text="Back", command=self.show_login).pack()

    def show_login(self):
        self.register_frame.pack_forget()
        self.login_frame.pack(pady=20, padx=40, fill="both", expand=True)

    def login(self):
        username = self.username.get()
        password = self.password.get()
        if self.app.user_db.verify_user(username, password):
            self.app.current_user = username
            self.frame.pack_forget()
            self.app.show_main_interface()
        else:
            messagebox.showerror("Error", "Invalid username or password")

    def register(self):
        username = self.username.get()
        password = self.password.get()
        email = self.email.get()
        if not all([username, password, email]):
            messagebox.showerror("Error", "All fields are required")
            return
        if self.app.user_db.create_user(username, password, email):
            messagebox.showinfo("Success", "Registration successful! Please login")
            self.show_login()
        else:
            messagebox.showerror("Error", "Username already exists")


class ArticleSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Web Scrapper Summarizer")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)

        # Custom color scheme
        self.primary_color = "#2c3e50"
        self.secondary_color = "#3498db"
        self.accent_color = "#e74c3c"
        self.light_bg = "#ecf0f1"
        self.dark_text = "#2c3e50"

        # Database and user management
        self.user_db = UserDB()
        self.current_user = None
        self.processor = ArticleProcessor()

        # Initialize authentication window
        self.auth_window = AuthWindow(root, self)
        self.main_frame = None

    def show_main_interface(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10, fg_color=self.light_bg)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # User info and logout
        self.user_frame = ctk.CTkFrame(self.main_frame, fg_color=self.primary_color)
        self.user_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        ctk.CTkLabel(self.user_frame,
                     text=f"Welcome, {self.current_user}",
                     text_color="white").pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.user_frame,
                      text="Logout",
                      command=self.logout,
                      fg_color=self.accent_color,
                      hover_color="#c0392b").pack(side=tk.RIGHT, padx=10)

        # Header Section
        self.header_frame = ctk.CTkFrame(self.main_frame, corner_radius=8, fg_color=self.primary_color)
        self.header_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        ctk.CTkLabel(self.header_frame,
                     text="WEB SCRAPPER SUMMARIZER",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="white").pack(pady=10)

        # URL Input Section
        self.url_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)
        self.url_frame.pack(fill=tk.X, pady=5, padx=5)

        ctk.CTkLabel(self.url_frame,
                     text="Enter Article URL:",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        self.url_entry = ctk.CTkEntry(self.url_frame,
                                      placeholder_text="https://example.com/article",
                                      width=600,
                                      corner_radius=8,
                                      fg_color="white",
                                      text_color=self.dark_text)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Settings Section
        self.settings_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)
        self.settings_frame.pack(fill=tk.X, pady=10, padx=5)

        # Accuracy Level
        accuracy_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        accuracy_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(accuracy_frame,
                     text="Accuracy Level:",
                     text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        self.accuracy_var = tk.StringVar(value="medium")
        accuracy_options = [
            ("High", "high", "#27ae60"),
            ("Medium", "medium", "#f39c12"),
            ("Low", "low", "#e74c3c")
        ]

        for text, val, color in accuracy_options:
            ctk.CTkRadioButton(accuracy_frame,
                               text=text,
                               variable=self.accuracy_var,
                               value=val,
                               radiobutton_height=16,
                               radiobutton_width=16,
                               fg_color=color,
                               hover_color=color,
                               text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        # Summary Length
        length_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        length_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(length_frame,
                     text="Summary Length:",
                     text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        self.length_var = tk.StringVar(value="medium")
        length_options = [
            ("Short", "short", "#3498db"),
            ("Medium", "medium", "#9b59b6"),
            ("Long", "long", "#2c3e50")
        ]

        for text, val, color in length_options:
            ctk.CTkRadioButton(length_frame,
                               text=text,
                               variable=self.length_var,
                               value=val,
                               radiobutton_height=16,
                               radiobutton_width=16,
                               fg_color=color,
                               hover_color=color,
                               text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        # Format Style
        format_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        format_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(format_frame,
                     text="Format Style:",
                     text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        self.format_var = tk.StringVar(value="standard")
        formats = [
            ("Standard", "standard", "#3498db"),
            ("Bullet Points", "bullet", "#2ecc71"),
            ("Academic", "academic", "#e74c3c"),
            ("Simplified", "simplified", "#f39c12")
        ]

        for text, val, color in formats:
            ctk.CTkRadioButton(format_frame,
                               text=text,
                               variable=self.format_var,
                               value=val,
                               radiobutton_height=16,
                               radiobutton_width=16,
                               fg_color=color,
                               hover_color=color,
                               text_color=self.dark_text).pack(side=tk.LEFT, padx=5)

        # Action Buttons
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill=tk.X, pady=10, padx=5)

        button_options = {
            "corner_radius": 8,
            "height": 40,
            "font": ctk.CTkFont(size=13, weight="bold"),
            "width": 180
        }

        ctk.CTkButton(self.button_frame,
                      text="Summarize",
                      command=self.summarize_article,
                      fg_color=self.secondary_color,
                      hover_color="#2980b9",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Save as PDF",
                      command=self.save_as_pdf,
                      fg_color="#9b59b6",
                      hover_color="#8e44ad",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Listen to Summary",
                      command=self.listen_summary,
                      fg_color="#1abc9c",
                      hover_color="#16a085",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Extract Images",
                      command=self.extract_images,
                      fg_color="#e67e22",
                      hover_color="#d35400",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Download Everything",
                      command=self.download_everything,
                      fg_color="#2ecc71",
                      hover_color="#27ae60",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Take Screenshot",
                      command=self.capture_screenshot,
                      fg_color="#e67e22",
                      hover_color="#d35400",
                      **button_options).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(self.button_frame,
                      text="Clear",
                      command=self.clear_all,
                      fg_color=self.accent_color,
                      hover_color="#c0392b",
                      **button_options).pack(side=tk.RIGHT, padx=10)

        # Results Section
        self.results_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        # Notebook for multiple tabs
        self.notebook = ctk.CTkTabview(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Summary Tab
        self.summary_tab = self.notebook.add("Summary")
        self.summary_text = ctk.CTkTextbox(self.summary_tab,
                                           wrap=tk.WORD,
                                           font=ctk.CTkFont(size=13),
                                           corner_radius=8,
                                           fg_color="white",
                                           text_color=self.dark_text)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Metadata Tab
        self.metadata_tab = self.notebook.add("Metadata")
        self.metadata_text = ctk.CTkTextbox(self.metadata_tab,
                                            wrap=tk.WORD,
                                            font=ctk.CTkFont(size=13),
                                            corner_radius=8,
                                            fg_color="white",
                                            text_color=self.dark_text)
        self.metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Images Tab
        self.images_tab = self.notebook.add("Images")

        # Create a canvas with scrollbar for images
        self.images_canvas = tk.Canvas(self.images_tab, bg="white", highlightthickness=0)
        self.images_scrollbar = ctk.CTkScrollbar(self.images_tab,
                                                 orientation="vertical",
                                                 command=self.images_canvas.yview)
        self.images_scrollable_frame = ctk.CTkFrame(self.images_canvas, fg_color="white")

        self.images_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.images_canvas.configure(
                scrollregion=self.images_canvas.bbox("all")
            )
        )

        self.images_canvas.create_window((0, 0), window=self.images_scrollable_frame, anchor="nw")
        self.images_canvas.configure(yscrollcommand=self.images_scrollbar.set)

        self.images_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.images_scrollbar.pack(side="right", fill="y", padx=5, pady=5)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ctk.CTkLabel(self.main_frame,
                                       textvariable=self.status_var,
                                       height=30,
                                       corner_radius=0,
                                       fg_color=self.primary_color,
                                       text_color="white",
                                       font=ctk.CTkFont(size=12))
        self.status_bar.pack(fill=tk.X, pady=(5, 0))

        # Initialize variables for image widgets
        self.image_widgets = []
        self.image_references = []  # To keep references to images
        self.current_audio_path = None

    def logout(self):
        self.main_frame.pack_forget()
        self.clear_all()
        self.current_user = None
        self.auth_window.frame.pack(pady=50, padx=50, fill="both", expand=True)

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def clear_all(self):
        self.url_entry.delete(0, tk.END)
        self.summary_text.delete("1.0", tk.END)
        self.metadata_text.delete("1.0", tk.END)

        # Clear images
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets = []
        self.image_references = []

        self.processor.cleanup()
        self.set_status("Ready")

    def summarize_article(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        if not self.processor.is_valid_url(url):
            messagebox.showerror("Error", "Please enter a valid URL starting with http:// or https://")
            return

        self.set_status("Processing article...")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Processing... Please wait...")

        # Disable buttons during processing
        for button in self.button_frame.winfo_children():
            if button.cget("text") != "Clear":
                button.configure(state="disabled")

        # Get settings from UI
        accuracy_level = self.accuracy_var.get()
        summary_length = self.length_var.get()
        format_style = self.format_var.get()

        # Run in a separate thread to avoid freezing the GUI
        def worker():
            try:
                metadata, summary = self.processor.summarize_article(
                    url, accuracy_level, summary_length, format_style
                )

                # Update UI in the main thread
                self.root.after(0, lambda: self.display_results(metadata, summary))
                self.root.after(0, lambda: self.set_status("Summary completed"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to summarize article: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error occurred"))
                logger.error(f"Summarization error: {str(e)}", exc_info=True)
            finally:
                # Re-enable buttons
                self.root.after(0, lambda: [button.configure(state="normal")
                                            for button in self.button_frame.winfo_children()
                                            if button.cget("text") != "Clear"])

        threading.Thread(target=worker, daemon=True).start()

    def display_results(self, metadata, summary):
        # Display summary
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)

        # Display metadata
        self.metadata_text.delete("1.0", tk.END)
        metadata_str = f"Title: {metadata['title']}\n"
        metadata_str += f"Author: {metadata['author']}\n"
        metadata_str += f"Published: {metadata['publish_date']}\n"
        metadata_str += f"Word Count: {metadata['word_count']}\n"
        metadata_str += f"Reading Time: {metadata['reading_time']} minutes\n"
        metadata_str += f"Format: {metadata['format_style']}\n"
        metadata_str += f"Keywords: {', '.join(metadata['keywords'])}\n"
        metadata_str += f"Sentiment: {metadata['sentiment']['label']} (score: {metadata['sentiment']['value']:.2f})\n"
        metadata_str += f"Readability: {metadata['readability']['grade_level']} - {metadata['readability']['reading_ease']}\n\n"

        if metadata['facts']:
            metadata_str += "Key Facts:\n"
            for fact in metadata['facts'][:5]:
                metadata_str += f"• {fact}\n"

        self.metadata_text.insert(tk.END, metadata_str)

        # Display images
        if hasattr(self.processor, 'current_images'):
            self.display_images(self.processor.current_images)

    def display_images(self, image_urls):
        # Clear previous images
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets = []
        self.image_references = []

        # Create a temporary directory for images if it doesn't exist
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "article_summarizer_images")
        os.makedirs(temp_dir, exist_ok=True)

        if not image_urls:
            no_img_label = ctk.CTkLabel(self.images_scrollable_frame,
                                        text="No images found in the article",
                                        text_color=self.dark_text)
            no_img_label.pack(pady=20)
            self.image_widgets.append(no_img_label)
            return

        # Add a download all button
        download_all_btn = ctk.CTkButton(
            self.images_scrollable_frame,
            text=f"Download All Images ({len(image_urls)})",
            command=lambda: self.download_all_images(image_urls),
            corner_radius=8,
            height=35,
            fg_color="#3498db",
            hover_color="#2980b9"
        )
        download_all_btn.pack(pady=15)
        self.image_widgets.append(download_all_btn)

        # Create a progress label
        progress_label = ctk.CTkLabel(self.images_scrollable_frame,
                                      text="Loading images...",
                                      text_color=self.dark_text)
        progress_label.pack(pady=5)
        self.image_widgets.append(progress_label)
        self.root.update_idletasks()  # Force UI update

        successful_images = 0
        for i, img_url in enumerate(image_urls):
            if i >= 15:  # Limit to first 15 images to prevent UI overload
                more_label = ctk.CTkLabel(self.images_scrollable_frame,
                                          text=f"Plus {len(image_urls) - 15} more images... Use 'Download All' to get them.",
                                          text_color=self.dark_text)
                more_label.pack(pady=10)
                self.image_widgets.append(more_label)
                break

            try:
                # Update progress
                progress_label.configure(text=f"Loading image {i + 1} of {min(15, len(image_urls))}...")
                self.root.update_idletasks()  # Force UI update

                # Create a frame for each image with download button
                img_frame = ctk.CTkFrame(self.images_scrollable_frame,
                                         corner_radius=8,
                                         fg_color="white",
                                         border_width=1,
                                         border_color="#ddd")
                img_frame.pack(fill=tk.X, pady=10, padx=10)

                # Download image
                local_img_path = os.path.join(temp_dir, f"img_{i}.png")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                # Fix URL validation and error handling
                if not img_url.startswith(('http://', 'https://')):
                    # Try to fix relative URLs
                    if img_url.startswith('/'):
                        # Get the base URL
                        from urllib.parse import urlparse
                        parsed_url = urlparse(self.url_entry.get().strip())
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        img_url = base_url + img_url
                    else:
                        # Skip invalid URLs
                        continue

                response = requests.get(img_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    with open(local_img_path, "wb") as f:
                        f.write(response.content)

                    # Create image label
                    try:
                        pil_img = Image.open(local_img_path)

                        # Resize large images
                        max_width = 500
                        if pil_img.width > max_width:
                            ratio = max_width / pil_img.width
                            new_height = int(pil_img.height * ratio)
                            pil_img = pil_img.resize((max_width, new_height), Image.LANCZOS)

                        # Add image to frame
                        tk_img = ImageTk.PhotoImage(pil_img)
                        self.image_references.append(tk_img)  # Keep a reference to prevent garbage collection

                        img_label = tk.Label(img_frame, image=tk_img, bg="white")
                        img_label.pack(pady=5)
                        img_label.image = tk_img  # Keep another reference to be safe

                        # Add image info
                        info_text = f"Image {i + 1}: {pil_img.width}x{pil_img.height} pixels"
                        info_label = ctk.CTkLabel(img_frame, text=info_text, text_color=self.dark_text)
                        info_label.pack(pady=5)

                        # Add download button
                        download_btn = ctk.CTkButton(
                            img_frame,
                            text="Download Image",
                            command=lambda path=local_img_path, idx=i: self.download_file(path, f"image_{idx + 1}.png"),
                            corner_radius=8,
                            height=30,
                            fg_color=self.secondary_color,
                            hover_color="#2980b9"
                        )
                        download_btn.pack(pady=(0, 10))

                        # Add to widgets list
                        self.image_widgets.extend([img_frame, img_label, info_label, download_btn])
                        successful_images += 1

                    except Exception as e:
                        logger.error(f"Failed to process image {img_url}: {e}")
                        error_label = ctk.CTkLabel(img_frame, text=f"Failed to load image: {str(e)}",
                                                   text_color=self.accent_color)
                        error_label.pack(pady=5)
                        self.image_widgets.append(error_label)

            except Exception as e:
                logger.error(f"Failed to display image {img_url}: {e}")
                error_label = ctk.CTkLabel(self.images_scrollable_frame,
                                           text=f"Failed to load image {i + 1}: {str(e)}",
                                           text_color=self.accent_color)
                error_label.pack(pady=5)
                self.image_widgets.append(error_label)

        # Remove progress label
        progress_label.destroy()
        self.image_widgets.remove(progress_label)

        # Add summary label
        summary_label = ctk.CTkLabel(self.images_scrollable_frame,
                                     text=f"Successfully loaded {successful_images} out of {len(image_urls)} images",
                                     text_color=self.dark_text)
        summary_label.pack(pady=10)
        self.image_widgets.append(summary_label)

    def download_all_images(self, image_urls):
        folder_path = filedialog.askdirectory(title="Select folder to save all images")
        if not folder_path:
            return

        for i, img_url in enumerate(image_urls):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(img_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    file_path = os.path.join(folder_path, f"image_{i + 1}.png")
                    with open(file_path, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                logger.error(f"Failed to download image {i}: {e}")

        messagebox.showinfo("Success", f"Downloaded {len(image_urls)} images to {folder_path}")
        self.set_status(f"Images saved to {folder_path}")

    def download_file(self, source_path, suggested_filename):
        try:
            save_path = filedialog.asksaveasfilename(
                defaultextension=os.path.splitext(suggested_filename)[1],
                filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")],
                initialfile=suggested_filename
            )

            if save_path:
                import shutil
                shutil.copy(source_path, save_path)
                messagebox.showinfo("Success", f"File saved to {save_path}")
                self.set_status(f"Image saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            self.set_status("Error saving image")

    def save_as_pdf(self):
        if not hasattr(self.processor, 'current_summary') or not self.processor.current_summary:
            messagebox.showerror("Error", "No summary to save. Please summarize an article first.")
            return

        self.set_status("Generating PDF...")

        def worker():
            try:
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".pdf",
                    filetypes=[("PDF Files", "*.pdf")],
                    initialfile=f"{self.processor.sanitize_filename(self.processor.current_metadata['title'])}_summary.pdf"
                )

                if save_path:
                    self.processor.generate_pdf(save_path)
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"PDF saved to {save_path}"))
                    self.root.after(0, lambda: self.set_status("PDF saved successfully"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error generating PDF"))
                logger.error(f"PDF generation error: {str(e)}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()

    def listen_summary(self):
        if not hasattr(self.processor, 'current_summary') or not self.processor.current_summary:
            messagebox.showerror("Error", "No summary to play. Please summarize an article first.")
            return

        self.set_status("Generating audio...")

        def worker():
            try:
                audio_path = self.processor.generate_audio()
                self.current_audio_path = audio_path  # Store the path

                if audio_path:
                    # Play the audio
                    if os.name == 'nt':  # Windows
                        import winsound
                        winsound.PlaySound(audio_path, winsound.SND_FILENAME)
                    else:  # macOS and Linux
                        import subprocess
                        subprocess.run(["afplay" if sys.platform == "darwin" else "aplay", audio_path])

                    # Show download button after playback
                    self.root.after(0, self.show_audio_download_button)
                    self.root.after(0, lambda: self.set_status("Audio playback completed"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate audio"))
                    self.root.after(0, lambda: self.set_status("Error generating audio"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to play audio: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error playing audio"))
                logger.error(f"Audio playback error: {str(e)}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()

    def show_audio_download_button(self):
        """Show a button to download the generated audio"""
        if not self.current_audio_path:
            return

        # Create a frame for the download button
        audio_frame = ctk.CTkFrame(self.results_frame, corner_radius=8)
        audio_frame.pack(pady=10, padx=10, fill=tk.X)

        # Add download button
        download_btn = ctk.CTkButton(
            audio_frame,
            text="Download Audio Summary",
            command=self.download_audio,
            corner_radius=8,
            height=35,
            fg_color="#1abc9c",
            hover_color="#16a085"
        )
        download_btn.pack(pady=10)

        # Store reference to remove later
        if not hasattr(self, 'audio_widgets'):
            self.audio_widgets = []
        self.audio_widgets.append(audio_frame)

    def download_audio(self):
        """Handle audio file download"""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            messagebox.showerror("Error", "No audio file to download")
            return

        try:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp3",
                filetypes=[("MP3 Files", "*.mp3"), ("All Files", "*.*")],
                initialfile="article_summary.mp3"
            )

            if save_path:
                import shutil
                shutil.copy(self.current_audio_path, save_path)
                messagebox.showinfo("Success", f"Audio saved to {save_path}")
                self.set_status(f"Audio saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save audio: {str(e)}")
            self.set_status("Error saving audio")

    def capture_screenshot(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        if not self.processor.is_valid_url(url):
            messagebox.showerror("Error", "Please enter a valid URL starting with http:// or https://")
            return

        self.set_status("Capturing screenshot...")

        def worker():
            try:
                screenshot_path = self.processor.capture_full_page_screenshot(url)
                self.root.after(0, lambda: self.display_screenshot(screenshot_path))
                self.root.after(0, lambda: self.set_status("Screenshot captured. Use the download button to save it."))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to capture screenshot: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error capturing screenshot"))
                logger.error(f"Screenshot error: {str(e)}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()

    def display_screenshot(self, screenshot_path):
        # Clear previous images
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets = []
        self.image_references = []

        try:
            # Create frame for screenshot
            screenshot_frame = ctk.CTkFrame(self.images_scrollable_frame,
                                            corner_radius=8,
                                            fg_color="white",
                                            border_width=1,
                                            border_color="#ddd")
            screenshot_frame.pack(fill=tk.X, pady=10, padx=10)

            # Load and display the screenshot
            pil_img = Image.open(screenshot_path)

            # Resize large images to fit window
            max_width = 800
            if pil_img.width > max_width:
                ratio = max_width / pil_img.width
                new_height = int(pil_img.height * ratio)
                pil_img = pil_img.resize((max_width, new_height), Image.LANCZOS)

            # Convert to Tkinter image
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_references.append(tk_img)  # Keep reference

            # Create label with image (using standard tkinter Label)
            img_label = tk.Label(screenshot_frame,
                                 image=tk_img,
                                 bg="white",
                                 borderwidth=0)
            img_label.pack(pady=5)

            # Add image info
            info_text = f"Screenshot: {pil_img.width}x{pil_img.height} pixels"
            info_label = ctk.CTkLabel(screenshot_frame,
                                      text=info_text,
                                      text_color=self.dark_text)
            info_label.pack(pady=5)

            # Add download button
            download_btn = ctk.CTkButton(
                screenshot_frame,
                text="Download Screenshot",
                command=lambda: self.download_file(screenshot_path, "screenshot.png"),
                corner_radius=8,
                height=30,
                fg_color=self.secondary_color,
                hover_color="#2980b9"
            )
            download_btn.pack(pady=(0, 10))

            # Add widgets to list
            self.image_widgets.extend([screenshot_frame, img_label, info_label, download_btn])

            # Select the Images tab
            self.notebook.set("Images")

        except Exception as e:
            logger.error(f"Failed to display screenshot: {e}")
            error_label = ctk.CTkLabel(self.images_scrollable_frame,
                                       text=f"Failed to load screenshot: {str(e)}",
                                       text_color=self.accent_color)
            error_label.pack(pady=5)
            self.image_widgets.append(error_label)

    def extract_images(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        if not self.processor.is_valid_url(url):
            messagebox.showerror("Error", "Please enter a valid URL starting with http:// or https://")
            return

        self.set_status("Extracting images...")

        def worker():
            try:
                image_urls = self.processor.extract_images_from_url(url, max_images=100)
                self.root.after(0, lambda: self.display_images(image_urls))
                self.root.after(0, lambda: self.set_status("Images extracted"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to extract images: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error occurred"))
                logger.error(f"Image extraction error: {str(e)}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()

    def download_everything(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        if not self.processor.is_valid_url(url):
            messagebox.showerror("Error", "Please enter a valid URL starting with http:// or https://")
            return

        self.set_status("Downloading everything...")

        def worker(self):
            try:
                save_path = filedialog.askdirectory(title="Select folder to save everything")
                if not save_path:
                    self.root.after(0, lambda: self.set_status("Download canceled"))
                    return

                self.root.after(0, lambda: self.set_status("Creating PDF summary..."))

                # Generate PDF
                pdf_filename = f"{self.processor.sanitize_filename(self.processor.current_metadata['title'])}_summary.pdf"
                pdf_path = os.path.join(save_path, pdf_filename)
                self.processor.generate_pdf(pdf_path)

                self.root.after(0, lambda: self.set_status("Creating audio summary..."))

                # Generate audio
                audio_path = os.path.join(save_path, "article_summary.mp3")
                self.processor.generate_audio(audio_path)

                # Download images
                if hasattr(self.processor, 'current_images') and self.processor.current_images:
                    self.root.after(0, lambda: self.set_status(
                        f"Downloading {len(self.processor.current_images)} images..."))
                    images_folder = os.path.join(save_path, "images")
                    os.makedirs(images_folder, exist_ok=True)

                    for i, img_url in enumerate(self.processor.current_images):
                        try:
                            headers = {'User-Agent': 'Mozilla/5.0'}
                            img_response = requests.get(img_url, headers=headers, timeout=30)
                            if img_response.status_code == 200:
                                img_filename = f"image_{i + 1}.png"
                                img_path = os.path.join(images_folder, img_filename)
                                with open(img_path, "wb") as f:
                                    f.write(img_response.content)
                        except Exception as e:
                            logger.error(f"Failed to download image {img_url}: {e}")

                self.root.after(0, lambda: messagebox.showinfo("Success", "All content downloaded successfully!"))
                self.root.after(0, lambda: self.set_status("Download completed"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to download content: {str(e)}"))
                self.root.after(0, lambda: self.set_status("Error occurred"))
                logger.error(f"Download error: {str(e)}", exc_info=True)

        # Start the worker thread
        threading.Thread(target=lambda: worker(self), daemon=True).start()


if __name__ == "__main__":
    root = ctk.CTk()
    app = ArticleSummarizerApp(root)
    root.mainloop()