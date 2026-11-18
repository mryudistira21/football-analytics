"""Unduh video pertandingan tanpa terminal — a small window for downloading.

Double-click "Unduh Video.bat" in the repo root to open this (no terminal
typing). It wraps src/download_match.py: 720p H.264 at 30 fps, merged, saved
to your video folder, and it can grab the whole match or just one half.

Half downloads cut at the nearest keyframe (no re-encoding, so they're fast and
light on the CPU). Enter the half's start and end as mm:ss, read from the video.
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_match as dm  # shared logic: format_selector, js_runtimes, etc.

MODES = [
    ("Full Pertandingan", "full"),
    ("Babak Pertama", "first"),
    ("Babak Kedua", "second"),
    ("Extra Time 1", "et1"),
    ("Extra Time 2", "et2"),
]
SUFFIX = {"full": "", "first": "_babak1", "second": "_babak2", "et1": "_et1", "et2": "_et2"}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.msgs: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        root.title("Unduh Video Pertandingan")
        root.minsize(620, 560)

        pad = dict(padx=10, pady=4)
        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(frm, text="Link YouTube").grid(row=row, column=0, sticky="w", **pad)
        self.url = ttk.Entry(frm)
        self.url.grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
        row += 1

        ttk.Label(frm, text="Folder simpan").grid(row=row, column=0, sticky="w", **pad)
        self.folder = ttk.Entry(frm)
        self.folder.insert(0, str(dm.DEFAULT_OUT))
        self.folder.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Pilih…", command=self.pick_folder).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(frm, text="Kualitas").grid(row=row, column=0, sticky="w", **pad)
        self.quality = ttk.Combobox(frm, values=["480p", "720p", "1080p"], state="readonly", width=10)
        self.quality.set("720p")
        self.quality.grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="Bagian").grid(row=row, column=0, sticky="nw", **pad)
        modes = ttk.Frame(frm)
        modes.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        self.mode = tk.StringVar(value="full")
        for i, (label, val) in enumerate(MODES):
            ttk.Radiobutton(modes, text=label, value=val, variable=self.mode,
                            command=self.sync_mode).grid(row=0, column=i, padx=(0, 14))
        row += 1

        self.time_frame = ttk.Frame(frm)
        self.time_frame.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        ttk.Label(self.time_frame, text="Mulai (mm:ss)").grid(row=0, column=0, padx=(0, 4))
        self.start = ttk.Entry(self.time_frame, width=8)
        self.start.grid(row=0, column=1, padx=(0, 16))
        ttk.Label(self.time_frame, text="Selesai (mm:ss)").grid(row=0, column=2, padx=(0, 4))
        self.end = ttk.Entry(self.time_frame, width=8)
        self.end.grid(row=0, column=3)
        row += 1
        self.time_hint = ttk.Label(frm, foreground="#666",
                                   text="Isi waktu babak dari jam di video — termasuk injury time (mis. 00:00 sampai 50:00 untuk babak 1 penuh).")
        self.time_hint.grid(row=row, column=1, columnspan=2, sticky="w", padx=10)
        row += 1

        ttk.Label(frm, text="Nama file").grid(row=row, column=0, sticky="w", **pad)
        self.name = ttk.Entry(frm)
        self.name.grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
        row += 1

        self.btn = ttk.Button(frm, text="Unduh", command=self.start_download)
        self.btn.grid(row=row, column=1, sticky="w", **pad)
        self.status = ttk.Label(frm, text="Siap.", foreground="#1f7a4a")
        self.status.grid(row=row, column=2, sticky="e", **pad)
        row += 1

        self.log = tk.Text(frm, height=14, wrap="word", state="disabled",
                           font=("Consolas", 9), background="#111", foreground="#ddd")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=10, pady=(8, 4))
        frm.rowconfigure(row, weight=1)

        self.sync_mode()
        self.root.after(120, self.drain)

    # ---------- helpers ----------
    def pick_folder(self):
        d = filedialog.askdirectory(initialdir=self.folder.get() or ".")
        if d:
            self.folder.delete(0, "end")
            self.folder.insert(0, d)

    def sync_mode(self):
        half = self.mode.get() != "full"
        state = "normal" if half else "disabled"
        for w in (self.start, self.end):
            w.configure(state=state)
        self.time_hint.configure(foreground="#666" if half else "#bbb")

    def log_line(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "log":
                    self.log_line(str(payload))
                elif kind == "status":
                    self.status.configure(text=str(payload))
                elif kind == "done":
                    self.finish(payload)
        except queue.Empty:
            pass
        self.root.after(120, self.drain)

    # ---------- download ----------
    def start_download(self):
        if self.running:
            return
        url = self.url.get().strip()
        if not url:
            messagebox.showwarning("Link kosong", "Tempel link YouTube dulu.")
            return
        out_dir = Path(self.folder.get().strip() or dm.DEFAULT_OUT)
        height = int(self.quality.get().replace("p", ""))
        mode = self.mode.get()
        start = end = None
        if mode != "full":
            start = dm.to_seconds(self.start.get()) if self.start.get().strip() else 0.0
            try:
                end = dm.to_seconds(self.end.get()) if self.end.get().strip() else None
            except ValueError:
                messagebox.showwarning("Waktu salah", "Gunakan format mm:ss, mis. 17:00.")
                return
            if end is not None and end <= start:
                messagebox.showwarning("Waktu salah", "Selesai harus lebih besar dari Mulai.")
                return
        name = self.name.get().strip() or None

        self.running = True
        self.btn.configure(state="disabled")
        self.status.configure(text="Mengunduh…", foreground="#b4541a")
        mode_label = dict((v, l) for l, v in MODES)[mode]
        self.log_line(f"Mulai mengunduh {self.quality.get()} — {mode_label}…")
        t = threading.Thread(target=self.worker,
                             args=(url, out_dir, height, mode, start, end, name), daemon=True)
        t.start()

    def worker(self, url, out_dir, height, mode, start, end, name):
        def emit(text):
            self.msgs.put(("log", text))
        try:
            import yt_dlp
        except ImportError:
            self.msgs.put(("done", "yt-dlp belum terpasang. Jalankan: pip install -U \"yt-dlp[default]\""))
            return
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = (name or "%(title).80B") + SUFFIX[mode]
            opts = {
                "format": dm.format_selector(height, 30),
                "merge_output_format": "mp4",
                "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
                "restrictfilenames": name is None,
                "noplaylist": True,
                "progress_hooks": [self.hook],
            }
            rt = dm.js_runtimes()
            if rt:
                opts["js_runtimes"] = rt
            if mode != "full":
                from yt_dlp.utils import download_range_func
                opts["download_ranges"] = download_range_func(
                    None, [(start, end if end else float("inf"))])
                opts["force_keyframes_at_cuts"] = False  # keyframe cut = no re-encode
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            path = dm.resolve_output(info, out_dir, stem)
            if path and path.exists():
                self.msgs.put(("done", f"OK|{path}"))
            else:
                self.msgs.put(("done", "Selesai, tapi file akhir tidak ditemukan di folder."))
        except Exception as exc:  # surface the real reason in the log
            emit(f"ERROR: {exc}")
            self.msgs.put(("done", f"Gagal: {exc}"))

    def hook(self, d):
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            spd = d.get("_speed_str", "").strip()
            self.msgs.put(("status", f"Mengunduh {pct} {spd}"))
        elif d.get("status") == "finished":
            self.msgs.put(("log", "Menggabungkan video + audio…"))

    def finish(self, payload: object):
        self.running = False
        self.btn.configure(state="normal")
        text = str(payload)
        if text.startswith("OK|"):
            path = text[3:]
            self.status.configure(text="Selesai.", foreground="#1f7a4a")
            self.log_line(f"Tersimpan: {path}")
            self.log_line("Buka Pitchside Tagger → 'Choose video file' → pilih file ini.")
            messagebox.showinfo("Selesai", f"Video tersimpan:\n{path}")
        else:
            self.status.configure(text="Gagal.", foreground="#b4541a")
            self.log_line(text)
            messagebox.showerror("Gagal", text)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
