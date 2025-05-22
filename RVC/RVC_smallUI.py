import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from rvc_python.infer import RVCInference
import soundfile as sf
import numpy as np
import torch
from fairseq.data.dictionary import Dictionary
from pydub import AudioSegment
torch.serialization.add_safe_globals([Dictionary])

class VoiceConversionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Conversion GUI")
        self.root.geometry("600x600")

        # Default directories
        self.default_output_dir = r"C:\Users\QiXuan\Downloads\Audio Deepfakes"
        self.source_audio_dir = r"C:\Users\QiXuan\Downloads\Youtube\audio_wav_only"
        self.model_dir = r"C:\Users\QiXuan\Downloads\Audio Deepfake Model\RVC\actual_models\model"
        self.index_dir = r"C:\Users\QiXuan\Downloads\Audio Deepfake Model\RVC\actual_models\index"
        os.makedirs(self.default_output_dir, exist_ok=True)  # Ensure output directory exists

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_label = tk.Label(root, text=f"Device: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}")
        self.device_label.pack(pady=5)

        # File selection variables
        self.source_audio = tk.StringVar()
        self.model_path = tk.StringVar()
        self.file_index = tk.StringVar()
        self.output_audio = tk.StringVar(value="someone_toBiden.wav")  # Just filename
        self.csv_file = tk.StringVar(value=os.path.join(self.default_output_dir, "audio_list.csv"))
        self.audio_duration = 0

        # GUI elements for file selection
        tk.Label(root, text="Source Audio:").pack()
        tk.Entry(root, textvariable=self.source_audio, width=50).pack()
        tk.Button(root, text="Browse", command=self.browse_source).pack()

        tk.Label(root, text="Model Path (.pth):").pack()
        tk.Entry(root, textvariable=self.model_path, width=50).pack()
        tk.Button(root, text="Browse", command=self.browse_model).pack()

        tk.Label(root, text="Index File (.index):").pack()
        tk.Entry(root, textvariable=self.file_index, width=50).pack()
        tk.Button(root, text="Browse", command=self.browse_index).pack()

        tk.Label(root, text="Output Audio (.wav):").pack()
        tk.Entry(root, textvariable=self.output_audio, width=50).pack()
        tk.Button(root, text="Browse", command=self.browse_output).pack()

        tk.Label(root, text="CSV File:").pack()
        tk.Entry(root, textvariable=self.csv_file, width=50).pack()
        tk.Button(root, text="Browse", command=self.browse_csv).pack()

        # Trimming options
        self.trim_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Trim Audio", variable=self.trim_var, command=self.toggle_trim).pack(pady=5)

        self.trim_frame = tk.Frame(root)
        tk.Label(self.trim_frame, text="Start Time (s):").pack(side=tk.LEFT)
        self.start_time = tk.Entry(self.trim_frame, width=10)
        self.start_time.pack(side=tk.LEFT, padx=5)
        tk.Label(self.trim_frame, text="End Time (s):").pack(side=tk.LEFT)
        self.end_time = tk.Entry(self.trim_frame, width=10)
        self.end_time.pack(side=tk.LEFT, padx=5)
        self.duration_label = tk.Label(self.trim_frame, text="Duration: N/A")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        self.trim_frame.pack()
        self.trim_frame.pack_forget()  # Hide by default

        # Run button and status
        tk.Button(root, text="Run Voice Conversion", command=self.run_conversion).pack(pady=10)
        self.status = tk.Label(root, text="", wraplength=500)
        self.status.pack(pady=5)

    def browse_source(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")], initialdir=self.source_audio_dir)
        if file:
            self.source_audio.set(os.path.normpath(file))
            self.update_duration()

    def browse_model(self):
        file = filedialog.askopenfilename(filetypes=[("PTH Files", "*.pth")], initialdir=self.model_dir)
        if file:
            self.model_path.set(os.path.normpath(file))

    def browse_index(self):
        file = filedialog.askopenfilename(filetypes=[("Index Files", "*.index")], initialdir=self.index_dir)
        if file:
            self.file_index.set(os.path.normpath(file))

    def browse_output(self):
        file = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Files", "*.wav")], initialdir=self.default_output_dir, initialfile=self.output_audio.get())
        if file:
            self.output_audio.set(os.path.normpath(file))

    def browse_csv(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], initialdir=self.default_output_dir)
        if file:
            self.csv_file.set(os.path.normpath(file))

    def toggle_trim(self):
        if self.trim_var.get():
            self.trim_frame.pack()
            self.update_duration()
        else:
            self.trim_frame.pack_forget()
            self.duration_label.config(text="Duration: N/A")

    def update_duration(self):
        if self.source_audio.get():
            try:
                audio = AudioSegment.from_file(self.source_audio.get())
                self.audio_duration = len(audio) / 1000
                self.duration_label.config(text=f"Duration: {self.audio_duration:.2f} s")
            except Exception as e:
                self.status.config(text=f"Error loading audio: {e}")

    def run_conversion(self):
        # Validate inputs
        if not all([self.source_audio.get(), self.model_path.get(), self.file_index.get(), self.output_audio.get(), self.csv_file.get()]):
            messagebox.showerror("Error", "Please select all required files.")
            return

        # Handle output audio path
        output_audio_path = self.output_audio.get()
        if os.path.dirname(output_audio_path):
            output_dir = os.path.dirname(output_audio_path)
        else:
            output_dir = self.default_output_dir
            output_audio_path = os.path.join(self.default_output_dir, output_audio_path)

        input_audio = self.source_audio.get()
        trimmed_audio = os.path.join(output_dir, "trimmed_source.wav")

        if self.trim_var.get():
            try:
                start_time = float(self.start_time.get())
                end_time = float(self.end_time.get())
                if not (0 <= start_time < end_time <= self.audio_duration):
                    messagebox.showerror("Error", "Invalid trim times.")
                    return
                audio = AudioSegment.from_file(input_audio)
                trimmed = audio[start_time * 1000:end_time * 1000]
                os.makedirs(output_dir, exist_ok=True)
                trimmed.export(trimmed_audio, format="wav")
                input_audio = trimmed_audio
                self.status.config(text=f"Trimmed audio saved to {trimmed_audio}")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for trim times.")
                return
            except Exception as e:
                messagebox.showerror("Error", f"Trimming failed: {e}")
                return

        # Run RVC inference
        try:
            rvc = RVCInference(model_path=self.model_path.get(), device=self.device, version="v2")
            result = rvc.vc.vc_single(
                sid=0,
                input_audio_path=input_audio,
                f0_up_key=0,
                f0_file=None,
                f0_method="harvest",
                file_index=self.file_index.get(),
                file_index2="",
                index_rate=0.6,
                filter_radius=0,
                resample_sr=0,
                rms_mix_rate=0.2,
                protect=0.15
            )
            if isinstance(result, tuple) and len(result) == 2:
                info, audio_opt = result
                if audio_opt[0] is None:
                    messagebox.showerror("Error", f"Conversion failed: {info}")
                    return
                audio_data = audio_opt[1]
            else:
                audio_data = result

            # Save output
            tgt_sr = rvc.vc.tgt_sr
            os.makedirs(output_dir, exist_ok=True)
            sf.write(output_audio_path, audio_data, tgt_sr)
            self.status.config(text=f"Voice conversion complete! Output saved as {output_audio_path}")

            # Update CSV
            metadata = {
                "filename": os.path.basename(output_audio_path),
                "deepfake_method": "VC",
                "model_used": "RVC",
                "original_audio": os.path.basename(self.source_audio.get()),
                "label": "spoof",
            }
            try:
                csv_dir = os.path.dirname(self.csv_file.get()) or self.default_output_dir
                os.makedirs(csv_dir, exist_ok=True)
                with open(self.csv_file.get(), mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=metadata.keys())
                    if os.stat(self.csv_file.get()).st_size == 0:
                        writer.writeheader()
                    writer.writerow(metadata)
                self.status.config(text=self.status.cget("text") + f"\nMetadata appended to {self.csv_file.get()}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update CSV: {e}")
                return

            # Open output (Windows only)
            if os.name == "nt":
                os.system(f'start "" "{output_audio_path}"')

            # Clean up
            if self.trim_var.get() and os.path.exists(trimmed_audio):
                os.remove(trimmed_audio)
                self.status.config(text=self.status.cget("text") + f"\nCleaned up: {trimmed_audio}")

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceConversionGUI(root)
    root.mainloop()