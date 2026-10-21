"""EDITH - a tiny Jarvis/EDITH-style HUD for a free voice assistant.

Run:  python assistant_interface.py
Needs assistant_core.py next to it and a .env file (see .env.example).
tkinter ships with the normal Windows Python installer.
"""
import math
import queue
import time
import tkinter as tk

import assistant_core as core

NAME = core.ASSISTANT_NAME.upper()

BG = "#050a12"
PANEL = "#08121f"
CYAN = "#00e5ff"
DIM = "#0b5560"
AMBER = "#ffb300"
RED = "#ff3d57"
VIOLET = "#9b7bff"
MINT = "#2dffb2"
TEXT = "#bff6ff"
FONT = "Consolas"


class HudApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.engine = core.Assistant(emit=lambda kind, text: self.q.put((kind, text)))
        self.error_until = 0.0
        self.angle = 0.0
        self.tick = 0

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.geometry("340x560+120+80")
        root.configure(bg=BG, highlightbackground=CYAN, highlightthickness=1)

        bar = tk.Frame(root, bg=PANEL)
        bar.pack(fill="x")
        title = tk.Label(bar, text=f"◈ {NAME}  //  A.I. INTERFACE", bg=PANEL, fg=CYAN,
                         font=(FONT, 9, "bold"), pady=6, padx=8)
        title.pack(side="left")
        close = tk.Label(bar, text="✕", bg=PANEL, fg=RED, font=(FONT, 11, "bold"),
                         padx=10, cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: self.quit())
        for w in (bar, title):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

        self.canvas = tk.Canvas(root, width=338, height=230, bg=BG, highlightthickness=0)
        self.canvas.pack()

        self.status = tk.Label(root, text="OFFLINE", bg=BG, fg=DIM, font=(FONT, 11, "bold"))
        self.status.pack()

        row = tk.Frame(root, bg=BG)
        row.pack(pady=10)
        self.btn_start = self._button(row, "▶ START", self.start, CYAN)
        self.btn_mute = self._button(row, "🎤 MUTE", self.toggle_mute, AMBER)
        self.btn_stop = self._button(row, "■ END", self.stop, RED)
        for b in (self.btn_start, self.btn_mute, self.btn_stop):
            b.pack(side="left", padx=4)

        tk.Label(root, text=f"Created by {core.CREATOR_NAME}", bg=BG, fg=DIM,
                 font=(FONT, 8)).pack(side="bottom", pady=(0, 4))

        frame = tk.Frame(root, bg=PANEL, highlightbackground=DIM, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(frame, bg=PANEL, fg=TEXT, font=(FONT, 9), wrap="word",
                           bd=0, padx=8, pady=6, state="disabled", height=8)
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("agent", foreground=CYAN)
        self.log.tag_config("user", foreground="#ffffff")
        self.log.tag_config("sys", foreground=AMBER)
        self.log.tag_config("err", foreground=RED)

        self._say("sys", f"SYSTEM: {NAME} offline. Initiate Neural Module. Press START.")
        self._animate()
        self._poll()

    def _button(self, parent, text, cmd, colour):
        b = tk.Label(parent, text=text, bg=BG, fg=colour, font=(FONT, 9, "bold"),
                     padx=10, pady=5, cursor="hand2",
                     highlightbackground=colour, highlightthickness=1)
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.config(bg=PANEL))
        b.bind("<Leave>", lambda e: b.config(bg=BG))
        return b

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y

    def _drag_move(self, e):
        x = self.root.winfo_x() + e.x - self._dx
        y = self.root.winfo_y() + e.y - self._dy
        self.root.geometry(f"+{x}+{y}")

    def _say(self, tag, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def start(self):
        self.btn_mute.config(text="🎤 MUTE")
        if not self.engine.running:
            self.log.config(state="normal")
            self.log.delete("1.0", "end")       
            self.log.config(state="disabled")
        self.engine.start()

    def stop(self):
        self.engine.stop()

    def toggle_mute(self):
        if not self.engine.running:
            return
        self.engine.muted = not self.engine.muted
        self.btn_mute.config(text="🎤 UNMUTE" if self.engine.muted else "🎤 MUTE")

    def quit(self):
        self.engine.stop()
        self.root.after(200, self.root.destroy)

    def _poll(self):
        try:
            while True:
                kind, text = self.q.get_nowait()
                if kind == "agent":
                    self._say("agent", f"{NAME}: {text}")
                elif kind == "user":
                    self._say("user", f"YOU: {text}")
                elif kind == "sys":
                    self._say("sys", f"SYSTEM: {text}")
                elif kind == "err":
                    self.error_until = time.time() + 4
                    self._say("err", f"ERROR: {text}")
                elif kind == "ended":
                    self.btn_mute.config(text="🎤 MUTE")
                    self._say("sys", "SYSTEM: Session closed.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _look(self):
        """(status text, colour) for the current engine state."""
        e = self.engine
        s = e.state
        if time.time() < self.error_until and s == "offline":
            return "ERROR", RED
        if s == "offline":
            return "SYSTEM OFFLINE", DIM
        if s == "limit":
            left = core.fmt_duration(e.blocked_until - time.time())
            return f"LIMIT REACHED - BACK IN {left}", AMBER
        if e.muted:
            return "MIC MUTED", AMBER
        return {
            "listening": ("LISTENING...", CYAN),
            "hearing": ("HEARING YOU", CYAN),
            "thinking": ("THINKING...", VIOLET),
            "speaking": (f"{NAME} SPEAKING", MINT),
        }.get(s, (s.upper(), CYAN))

    def _animate(self):
        c = self.canvas
        c.delete("all")
        self.tick += 1
        cx, cy = 169, 115
        e = self.engine
        text, col = self._look()
        self.status.config(text=text, fg=col)

        fast = e.state in ("hearing", "thinking", "speaking") and not e.muted
        self.angle += 3.4 if fast else 0.8

        c.create_oval(cx - 100, cy - 100, cx + 100, cy + 100, outline=col, width=2)
        for i in range(36):
            a = math.radians(i * 10)
            r1, r2 = 92, 100 if i % 3 else 108
            c.create_line(cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                          cx + r2 * math.cos(a), cy + r2 * math.sin(a), fill=col)
        for k in range(3):
            c.create_arc(cx - 78, cy - 78, cx + 78, cy + 78,
                         start=self.angle + k * 120, extent=70,
                         style="arc", outline=col, width=4)
        c.create_arc(cx - 58, cy - 58, cx + 58, cy + 58,
                     start=-self.angle * 1.6, extent=250, style="arc", outline=col, width=2)

        if e.state == "speaking":
            pulse = 5 + 4 * math.sin(self.tick / 3)
        else:
            pulse = e.level * 18
            e.level *= 0.8
        r = 24 + pulse
        c.create_oval(cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8, outline=col, width=1)
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col, outline="")
        c.create_oval(cx - r / 2, cy - r / 2, cx + r / 2, cy + r / 2, fill=BG, outline="")

        self.root.after(40, self._animate)


if __name__ == "__main__":
    root = tk.Tk()
    HudApp(root)
    root.mainloop()