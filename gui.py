from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk

from services.export_service import save_all_outputs
from services.transcription_service import TranscriptionService, WHISPER_MODELS


class TranskriptorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Transkriptor")
        self.root.geometry("900x650")
        self.root.minsize(800, 550)

        self.service = TranscriptionService()
        self.queue: Queue = Queue()

        self.selected_file = tk.StringVar()
        self.model_name = tk.StringVar(value="small")
        self.language = tk.StringVar(value="pt")
        self.status_text = tk.StringVar(value="Ready")

        self.current_result: dict | None = None
        self.text_output: scrolledtext.ScrolledText | None = None
        self.transcribe_button: ttk.Button | None = None
        self.save_button: ttk.Button | None = None
        self.progress_bar: ttk.Progressbar | None = None
        self.progress_value = tk.DoubleVar(value=0.0)
        self.progress_label_text = tk.StringVar(value="0%")
        self.progress_label: ttk.Label | None = None

        self._build_layout()
        self._poll_queue()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        container.columnconfigure(1, weight=1)
        container.rowconfigure(6, weight=1)

        ttk.Label(container, text="Input file").grid(
            row=0, column=0, sticky="w", pady=5
        )

        ttk.Entry(container, textvariable=self.selected_file).grid(
            row=0, column=1, sticky="ew", padx=5
        )

        ttk.Button(container, text="Browse", command=self._browse_file).grid(
            row=0, column=2, padx=(5, 0)
        )

        ttk.Label(container, text="Model").grid(
            row=1, column=0, sticky="w", pady=5
        )

        ttk.Combobox(
            container,
            textvariable=self.model_name,
            values=WHISPER_MODELS,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(container, text="Language").grid(
            row=2, column=0, sticky="w", pady=5
        )

        ttk.Entry(container, textvariable=self.language, width=10).grid(
            row=2, column=1, sticky="w", padx=5
        )

        button_frame = ttk.Frame(container)
        button_frame.grid(row=3, column=0, columnspan=3, pady=12, sticky="w")

        self.transcribe_button = ttk.Button(
            button_frame,
            text="Transcribe",
            command=self._start_transcription,
        )
        self.transcribe_button.pack(side="left", padx=(0, 8))

        self.save_button = ttk.Button(
            button_frame,
            text="Save transcription",
            command=self._save_transcription,
            state="disabled",
        )
        self.save_button.pack(side="left")

        ttk.Label(container, textvariable=self.status_text).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )

        self.progress_bar = ttk.Progressbar(
            container,
            mode="determinate",
            maximum=100,
            variable=self.progress_value,
        )
        self.progress_bar.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10)
        )

        self.progress_label = ttk.Label(container, textvariable=self.progress_label_text)
        self.progress_label.grid(
            row=5, column=2, sticky="e", pady=(0, 10)
        )

        self.text_output = scrolledtext.ScrolledText(
            container,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
        )
        self.text_output.grid(
            row=6, column=0, columnspan=3, sticky="nsew"
        )

    def _browse_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=[
                ("Media files", "*.mp3 *.wav *.m4a *.mp4 *.mpeg *.mpga *.webm"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.selected_file.set(file_path)

    def _start_transcription(self) -> None:
        input_path = self.selected_file.get().strip()
        language = self.language.get().strip()
        model_name = self.model_name.get().strip()

        if not input_path:
            messagebox.showwarning("Validation", "Please select an input file.")
            return

        if not language:
            messagebox.showwarning("Validation", "Please provide a language code.")
            return

        self.current_result = None

        if self.text_output is not None:
            self.text_output.delete("1.0", tk.END)

        if self.transcribe_button is not None:
            self.transcribe_button.config(state="disabled")

        if self.save_button is not None:
            self.save_button.config(state="disabled")

        if self.progress_bar is not None:
            self.progress_bar.stop()

        self.progress_value.set(0)
        self.progress_label_text.set("100%")
        self.status_text.set("Starting transcription...")

        worker = threading.Thread(
            target=self._run_transcription,
            args=(input_path, model_name, language),
            daemon=True,
        )
        worker.start()

    def _run_transcription(
            self,
            input_path: str,
            model_name: str,
            language: str,
    ) -> None:
        try:
            result = self.service.transcribe_file_with_progress(
                input_file=Path(input_path),
                model_name=model_name,
                language=language,
                chunk_duration=30,
                progress_callback=self._on_transcription_progress,
            )

            self.queue.put(
                {
                    "type": "success",
                    "result": result,
                }
            )
        except Exception as exc:
            self.queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

    def _save_transcription(self) -> None:
        if self.current_result is None:
            messagebox.showwarning(
                "Save transcription",
                "There is no transcription available to save.",
            )
            return

        input_path = self.selected_file.get().strip()
        base_name = Path(input_path).stem if input_path else "transcription"

        output_dir = filedialog.askdirectory(title="Select output folder")
        if not output_dir:
            return

        try:
            paths = save_all_outputs(
                output_dir=Path(output_dir),
                base_name=base_name,
                result=self.current_result,
            )

            self.status_text.set(
                f"Saved successfully. TXT: {paths['txt']} | SRT: {paths['srt']} | JSON: {paths['json']}"
            )

            messagebox.showinfo(
                "Save transcription",
                "Transcription files were saved successfully.",
            )
        except Exception as exc:
            messagebox.showerror(
                "Save transcription error",
                str(exc),
            )

    def _on_transcription_progress(self, percentage: float, message: str) -> None:
        self.queue.put(
            {
                "type": "progress",
                "percentage": percentage,
                "message": message,
            }
        )

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.queue.get_nowait()

                if message["type"] == "progress":
                    percentage = float(message["percentage"])
                    progress_message = message["message"]

                    self.progress_value.set(percentage)
                    self.progress_label_text.set(f"{percentage:.0f}%")
                    self.status_text.set(progress_message)

                elif message["type"] == "success":
                    self.current_result = message["result"]

                    if self.text_output is not None:
                        self.text_output.delete("1.0", tk.END)
                        self.text_output.insert("1.0", self.current_result["text"].strip())

                    if self.progress_bar is not None:
                        self.progress_bar.stop()

                    self.progress_value.set(100)
                    self.progress_label_text.set("100%")
                    self.status_text.set("Transcription completed. You can now review and save the result.")

                    if self.transcribe_button is not None:
                        self.transcribe_button.config(state="normal")

                    if self.save_button is not None:
                        self.save_button.config(state="normal")

                elif message["type"] == "error":
                    self.status_text.set("Error during transcription.")

                    if self.progress_bar is not None:
                        self.progress_bar.stop()

                    if self.transcribe_button is not None:
                        self.transcribe_button.config(state="normal")

                    if self.save_button is not None:
                        self.save_button.config(state="disabled")

                    messagebox.showerror(
                        "Transcription error",
                        message["message"],
                    )

        except Empty:
            pass

        self.root.after(200, self._poll_queue)