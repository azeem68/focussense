
"""
Front_1.py — FocusSense Desktop UI
=====================================
PyQt5 frontend for the attention tracking system.

Architecture:
  Test_1.py  →  trains models  (run once)
  main.py    →  AttentionEngine class  (ML + CV backend)
  Front_1.py →  THIS FILE — Qt UI + InferenceWorker + warning system

Run:
    python Front_1.py

The InferenceWorker QThread:
  • Imports AttentionEngine from main.py
  • Opens webcam, feeds frames into engine.process()
  • Emits (annotated_frame, state_dict) to the GUI thread every ~30 ms

Warning system:
  • DistractionWarningEngine counts continuous distraction seconds
  • Every WARNING_INTERVAL_S (15s) → full-screen red flash + popup + OS tray beep
  • Resets instantly when focus is restored
"""

import sys, time, math, threading
import ssl, certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import cv2
import numpy as np

# ── import the entire backend from main.py ──────────────────────────────────
from main import (
    AttentionEngine,
    WARNING_INTERVAL_S,
    MAX_DISTANCE_CM,
)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QTabWidget, QTextEdit, QSystemTrayIcon, QMenu, QAction,
    QSizePolicy,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QTime,
    QPropertyAnimation, QEasingCurve, QPoint, QRect, QRectF,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen,
    QBrush, QPalette, QPolygon, QRadialGradient,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Black + Green palette
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg0" : "#050d07",
    "bg1" : "#0a140c",
    "bg2" : "#0f1f11",
    "bg3" : "#162a18",
    "grn1": "#16a34a",
    "grn2": "#22c55e",
    "grn3": "#4ade80",
    "grn4": "#bbf7d0",
    "grn5": "#14532d",
    "grn6": "#052e16",
    "red" : "#ef4444",
    "yel" : "#f59e0b",
    "txt" : "#dcfce7",
    "txt2": "#86efac",
    "txt3": "#4ade80",
    "muted":"#166534",
    "bdr" : "#162a18",
    "bdr2": "#14532d",
}

QSS = f"""
QMainWindow, QWidget#root {{
    background:{C['bg0']}; color:{C['txt']};
    font-family:'Courier New',monospace; }}
QWidget {{
    background:transparent; color:{C['txt']};
    font-family:'Courier New',monospace; font-size:12px; }}
QFrame#card {{
    background:{C['bg1']}; border:1px solid {C['bdr2']};
    border-radius:12px; }}
QFrame#card_inner {{
    background:{C['bg2']}; border:1px solid {C['bdr']};
    border-radius:8px; }}
QPushButton {{
    background:{C['bg2']}; color:{C['txt']};
    border:1px solid {C['bdr2']}; border-radius:8px;
    padding:8px 18px; font-family:'Courier New',monospace;
    font-size:11px; letter-spacing:1px; }}
QPushButton:hover {{
    background:{C['bg3']}; border-color:{C['grn1']}; color:{C['grn3']}; }}
QPushButton:pressed {{ background:{C['grn5']}; }}
QPushButton#btn_primary {{
    background:{C['grn5']}; border-color:{C['grn2']};
    color:{C['grn4']}; font-weight:bold; }}
QPushButton#btn_primary:hover {{ background:{C['grn1']}; }}
QPushButton#btn_danger {{
    background:rgba(239,68,68,30); border-color:rgba(239,68,68,120);
    color:{C['red']}; }}
QPushButton#btn_danger:hover {{ background:rgba(239,68,68,60); }}
QPushButton#btn_warn_dismiss {{
    background:rgba(239,68,68,40); border:2px solid {C['red']};
    border-radius:10px; color:{C['red']}; font-size:13px;
    font-weight:bold; padding:10px 28px; letter-spacing:2px; }}
QPushButton#btn_warn_dismiss:hover {{ background:rgba(239,68,68,90); }}
QProgressBar {{
    background:{C['bg3']}; border:none; border-radius:3px;
    height:6px; text-align:center; color:transparent; }}
QProgressBar::chunk  {{ border-radius:3px; background:{C['grn2']}; }}
QProgressBar#bar_green::chunk  {{ background:{C['grn2']}; }}
QProgressBar#bar_red::chunk    {{ background:{C['red']}; }}
QProgressBar#bar_yel::chunk    {{ background:{C['yel']}; }}
QProgressBar#bar_orange::chunk {{ background:#f97316; }}
QTabWidget::pane {{
    border:1px solid {C['bdr2']}; border-radius:0 8px 8px 8px;
    background:{C['bg1']}; }}
QTabBar::tab {{
    background:{C['bg2']}; color:{C['muted']};
    border:1px solid {C['bdr']}; border-bottom:none;
    padding:8px 16px; font-size:10px; letter-spacing:1px;
    margin-right:2px; border-radius:6px 6px 0 0; }}
QTabBar::tab:selected {{
    background:{C['bg1']}; color:{C['grn3']}; border-color:{C['bdr2']}; }}
QTabBar::tab:hover {{ color:{C['grn3']}; }}
QTextEdit {{
    background:{C['bg2']}; color:{C['grn3']};
    border:1px solid {C['bdr']}; border-radius:6px;
    font-family:'Courier New',monospace; font-size:11px; padding:6px; }}
QScrollBar:vertical {{
    background:{C['bg2']}; width:6px; border-radius:3px; }}
QScrollBar::handle:vertical {{
    background:{C['grn5']}; border-radius:3px; min-height:20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  InferenceWorker  —  runs AttentionEngine in a background QThread
# ══════════════════════════════════════════════════════════════════════════════
class InferenceWorker(QThread):
    """
    Opens the webcam, feeds each frame into AttentionEngine.process(),
    then emits the annotated frame + state dict to the GUI thread.
    """
    frame_ready   = pyqtSignal(np.ndarray, dict)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._paused  = False
        self._engine  = AttentionEngine()

    def run(self):
        # Load models in the worker thread to keep the UI responsive
        self._engine.load(log_fn=lambda m: self.status_update.emit(m))

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_update.emit("Webcam unavailable — demo frames only")
            cap = None

        self._running = True
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            if cap:
                ret, frame = cap.read()
                if not ret:
                    continue
            else:
                # Demo frame: black canvas
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # All ML inference happens here, inside main.py's AttentionEngine
            state = self._engine.process(frame)

            self.frame_ready.emit(frame, state)
            time.sleep(0.033)   # ~30 fps

        if cap:
            cap.release()

    def toggle_pause(self):
        self._paused = not self._paused

    def stop(self):
        self._running = False
        self.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  DistractionWarningEngine  —  15-second continuous distraction counter
# ══════════════════════════════════════════════════════════════════════════════
class DistractionWarningEngine(QThread):
    warn_fire   = pyqtSignal(int, int)   # (warn_number, distract_secs)
    tick_update = pyqtSignal(int, int)   # (distract_secs, pct 0-100)

    def __init__(self):
        super().__init__()
        self._lock          = threading.Lock()
        self._focused       = True
        self._running       = False
        self._distract_secs = 0
        self._warn_count    = 0

    def set_focused(self, focused: bool):
        with self._lock:
            if focused and not self._focused:
                self._distract_secs = 0   # instant reset on focus return
            self._focused = focused

    def run(self):
        self._running = True
        while self._running:
            time.sleep(1.0)
            with self._lock:
                focused = self._focused
            if not focused:
                self._distract_secs += 1
                pct = int((self._distract_secs % WARNING_INTERVAL_S)
                          / WARNING_INTERVAL_S * 100)
                self.tick_update.emit(self._distract_secs, pct)
                if self._distract_secs % WARNING_INTERVAL_S == 0:
                    self._warn_count += 1
                    self.warn_fire.emit(self._warn_count, self._distract_secs)
            else:
                self._distract_secs = 0
                self.tick_update.emit(0, 0)

    def stop(self):
        self._running = False
        self.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  FullScreenRedFlash  —  covers entire primary display
# ══════════════════════════════════════════════════════════════════════════════
class FullScreenRedFlash(QWidget):
    def __init__(self):
        super().__init__(None,
                         Qt.FramelessWindowHint |
                         Qt.WindowStaysOnTopHint |
                         Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setVisible(False)
        self._alpha  = 0
        self._phase  = 0
        self._pulse  = 0.0
        self._warn_n = 0
        self._secs   = 0

        self._tick = QTimer(self); self._tick.timeout.connect(self._anim_tick)
        self._hide = QTimer(self, singleShot=True)
        self._hide.timeout.connect(self._begin_fadeout)

        self._dismiss_btn = QPushButton("DISMISS  ✕", self)
        self._dismiss_btn.setStyleSheet(
            "QPushButton{background:rgba(239,68,68,80);border:2px solid #ef4444;"
            "border-radius:12px;color:white;font-size:14px;font-weight:bold;"
            "padding:12px 36px;letter-spacing:2px;}"
            "QPushButton:hover{background:rgba(239,68,68,140);}")
        self._dismiss_btn.clicked.connect(self._begin_fadeout)

    def flash(self, warn_n: int, secs: int):
        self._warn_n = warn_n; self._secs = secs
        self._alpha  = 0; self._phase = 0; self._pulse = 0.0
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self._dismiss_btn.move(screen.width()//2 - 100, screen.height() - 110)
        self.show(); self.raise_()
        self._tick.start(30); self._hide.start(5000)

    def _begin_fadeout(self):
        self._hide.stop(); self._phase = 2

    def _anim_tick(self):
        if self._phase == 0:
            self._alpha = min(self._alpha + 15, 210)
            if self._alpha >= 210: self._phase = 1
        elif self._phase == 1:
            self._pulse = (self._pulse + 0.14) % (2 * math.pi)
        elif self._phase == 2:
            self._alpha = max(self._alpha - 20, 0)
            if self._alpha == 0:
                self._tick.stop(); self.hide()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        ba = min(self._alpha + (int(abs(math.sin(self._pulse))*50) if self._phase==1 else 0), 255)

        p.fillRect(0, 0, W, H, QColor(200, 0, 0, min(self._alpha//2, 110)))
        p.setPen(QPen(QColor(239, 68, 68, ba), 10))
        p.drawRect(5, 5, W-10, H-10)

        # Central warning box
        bw, bh = 700, 195
        bx, by = (W-bw)//2, (H-bh)//2 - 25
        p.setBrush(QBrush(QColor(100, 0, 0, min(ba, 230))))
        p.setPen(QPen(QColor(239, 68, 68, ba), 3))
        p.drawRoundedRect(bx, by, bw, bh, 16, 16)

        p.setPen(QColor(255, 255, 255, min(ba+40, 255)))
        p.setFont(QFont("Courier New", 26, QFont.Bold))
        p.drawText(QRect(bx, by+18, bw, 58), Qt.AlignCenter,
                   f"⚠  FOCUS ALERT  #{self._warn_n}")
        p.setFont(QFont("Courier New", 13))
        p.setPen(QColor(255, 180, 180, min(ba, 230)))
        p.drawText(QRect(bx, by+85, bw, 38), Qt.AlignCenter,
                   f"You've been distracted for {self._secs} seconds.")
        p.setFont(QFont("Courier New", 12))
        p.drawText(QRect(bx, by+128, bw, 34), Qt.AlignCenter,
                   "Return your gaze to the screen and stay productive.")

        p.setFont(QFont("Courier New", 9))
        p.setPen(QColor(255, 120, 120, min(ba-40, 180)))
        p.drawText(QRect(0, H-44, W, 28), Qt.AlignCenter,
                   "FocusSense — Eye gaze & head pose tracking")


# ══════════════════════════════════════════════════════════════════════════════
#  WarningPopup  —  slide-in card (top-right of main window)
# ══════════════════════════════════════════════════════════════════════════════
class WarningPopup(QWidget):
    dismissed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(375, 215)
        self._build()
        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _build(self):
        ol = QVBoxLayout(self); ol.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setStyleSheet(
            f"QFrame{{background:{C['bg1']};border:2px solid {C['red']};"
            f"border-radius:14px;}}")
        lay = QVBoxLayout(card); lay.setContentsMargins(20,16,20,16); lay.setSpacing(10)

        hdr = QHBoxLayout()
        ic  = QLabel("⚠"); ic.setStyleSheet(f"color:{C['red']};font-size:26px;")
        tc  = QVBoxLayout(); tc.setSpacing(2)
        self._title = QLabel("FOCUS ALERT")
        self._title.setStyleSheet(
            f"color:{C['red']};font-size:14px;font-weight:bold;letter-spacing:2px;")
        self._sub = QLabel("Warning #0")
        self._sub.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        tc.addWidget(self._title); tc.addWidget(self._sub)
        hdr.addWidget(ic); hdr.addLayout(tc); hdr.addStretch()

        self._msg = QLabel("Eyes detected off-screen.\nClose distracting apps and refocus.")
        self._msg.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        self._msg.setWordWrap(True)

        self._cd_lbl = QLabel("Auto-dismiss in 8s")
        self._cd_lbl.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        self._cd_bar = QProgressBar()
        self._cd_bar.setRange(0,8); self._cd_bar.setValue(8); self._cd_bar.setFixedHeight(4)
        self._cd_bar.setObjectName("bar_red"); self._cd_bar.setStyle(self._cd_bar.style())

        btn = QPushButton("  DISMISS  ✕"); btn.setObjectName("btn_warn_dismiss")
        btn.clicked.connect(self._dismiss)

        lay.addLayout(hdr); lay.addWidget(self._msg)
        lay.addWidget(self._cd_lbl); lay.addWidget(self._cd_bar)
        lay.addWidget(btn, alignment=Qt.AlignRight)
        ol.addWidget(card)

        self._cdv = 8
        self._cdt = QTimer(self); self._cdt.timeout.connect(self._cd_tick)

    def show_warning(self, n: int, secs: int):
        self._sub.setText(f"Warning #{n}  ·  {secs}s continuous distraction")
        self._msg.setText(
            f"Gaze detected off-screen for {secs}s.\n"
            "Close distracting apps and stay focused.")
        pw = self.parent().width(); ph = self.parent().height()
        tx = pw - self.width() - 18; ty = 75
        self.move(tx, -self.height())
        self.setVisible(True); self.raise_()
        self._anim.stop()
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(QPoint(tx, ty))
        self._anim.start()
        self._cdv = 8; self._cd_bar.setValue(8)
        self._cd_lbl.setText("Auto-dismiss in 8s")
        self._cdt.start(1000)

    def _cd_tick(self):
        self._cdv -= 1
        self._cd_bar.setValue(self._cdv)
        self._cd_lbl.setText(
            f"Auto-dismiss in {self._cdv}s" if self._cdv > 0 else "Dismissing…")
        if self._cdv <= 0: self._cdt.stop(); self._dismiss()

    def _dismiss(self):
        self._cdt.stop(); self.setVisible(False); self.dismissed.emit()


# ══════════════════════════════════════════════════════════════════════════════
#  GazeScreenWidget  —  miniature laptop screen with live gaze dot
# ══════════════════════════════════════════════════════════════════════════════
class GazeScreenWidget(QWidget):
    TRAIL_LEN = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 160)
        self._gx = 0.5; self._gy = 0.5
        self._focused = True
        self._trail = []
        self._label = "CENTER"

    def set_gaze(self, nx: float, ny: float, focused: bool):
        self._gx = nx; self._gy = ny; self._focused = focused
        self._trail.append((nx, ny))
        if len(self._trail) > self.TRAIL_LEN:
            self._trail.pop(0)
        col = nx; row = ny
        hz = "LEFT" if col < 0.33 else ("RIGHT" if col > 0.67 else "CENTER")
        vt = "TOP"  if row < 0.33 else ("BOTTOM" if row > 0.67 else "")
        self._label = f"{vt} {hz}".strip() if vt else hz
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # Screen bezel
        bx, by, bw, bh = 10, 8, W-20, H-22
        p.setPen(QPen(QColor(C['grn5']), 2))
        p.setBrush(QBrush(QColor(C['bg2'])))
        p.drawRoundedRect(bx, by, bw, bh, 6, 6)

        # Screen area
        sx, sy, sw, sh = bx+6, by+6, bw-12, bh-12
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(C['bg3'])))
        p.drawRoundedRect(sx, sy, sw, sh, 3, 3)

        # Grid
        p.setPen(QPen(QColor(C['grn5']), 1))
        for i in (1, 2):
            p.drawLine(int(sx+sw*i/3), sy, int(sx+sw*i/3), sy+sh)
            p.drawLine(sx, int(sy+sh*i/3), sx+sw, int(sy+sh*i/3))

        # Trail
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(255 * (i / max(len(self._trail)-1, 1)) * 0.3)
            r2 = max(1, int(3 * i / max(len(self._trail)-1, 1)))
            tc = QColor(C['grn2']); tc.setAlpha(alpha)
            p.setPen(Qt.NoPen); p.setBrush(QBrush(tc))
            p.drawEllipse(int(sx+tx*sw)-r2, int(sy+ty*sh)-r2, r2*2, r2*2)

        # Gaze dot
        dx = int(sx + self._gx * sw); dy = int(sy + self._gy * sh)
        dot_col = QColor(C['grn3']) if self._focused else QColor(C['red'])
        grad = QRadialGradient(dx, dy, 14)
        glow = QColor(dot_col); glow.setAlpha(55)
        grad.setColorAt(0, glow); grad.setColorAt(1, QColor(0,0,0,0))
        p.setBrush(QBrush(grad)); p.setPen(Qt.NoPen)
        p.drawEllipse(dx-14, dy-14, 28, 28)
        p.setBrush(QBrush(dot_col)); p.setPen(QPen(QColor(C['bg0']), 1))
        p.drawEllipse(dx-6, dy-6, 12, 12)
        p.setPen(QPen(dot_col, 1))
        p.drawLine(dx-12, dy, dx-7, dy); p.drawLine(dx+7, dy, dx+12, dy)
        p.drawLine(dx, dy-12, dx, dy-7); p.drawLine(dx, dy+7, dx, dy+12)

        # Zone label
        p.setPen(QPen(dot_col)); p.setFont(QFont("Courier New", 8, QFont.Bold))
        p.drawText(QRect(sx, sy+2, sw, 14), Qt.AlignCenter, self._label)

        # Laptop stand
        mx = W//2
        p.setPen(QPen(QColor(C['grn5']), 2))
        p.drawLine(mx-20, H-8, mx+20, H-8)
        p.drawLine(mx-6, by+bh, mx-20, H-8)
        p.drawLine(mx+6, by+bh, mx+20, H-8)

        # Coords
        p.setPen(QPen(QColor(C['muted']))); p.setFont(QFont("Courier New", 8))
        p.drawText(QRect(sx, sy+sh-14, sw, 14), Qt.AlignCenter,
                   f"x:{self._gx:.2f}  y:{self._gy:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Reusable mini-widgets
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  CameraLabel  —  QLabel with double-click/double-tap → full-screen camera view
# ══════════════════════════════════════════════════════════════════════════════
class FullScreenCameraWindow(QWidget):
    """Borderless full-screen overlay that mirrors the camera feed."""
    closed = pyqtSignal()

    def __init__(self):
        super().__init__(None,
                         Qt.FramelessWindowHint |
                         Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet(f"background:{C['bg0']};")
        self._pixmap = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("background:black;")
        lay.addWidget(self._lbl)

        # Hint label
        self._hint = QLabel("Double-tap / Double-click to exit full screen")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet(
            f"color:{C['muted']};font-size:11px;letter-spacing:1px;"
            f"padding:6px;background:{C['bg1']};")
        self._hint.setFixedHeight(30)
        lay.addWidget(self._hint)

        # Auto-hide hint after 3 s
        self._hint_timer = QTimer(self, singleShot=True)
        self._hint_timer.timeout.connect(lambda: self._hint.hide())

        # Double-click detection
        self._last_click = 0.0

    def show_fullscreen(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self._hint.show()
        self._hint_timer.start(3000)
        self.showFullScreen()
        self.raise_()

    def update_frame(self, pixmap: QPixmap):
        if not self.isVisible():
            return
        scaled = pixmap.scaled(
            self._lbl.width(), self._lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._lbl.setPixmap(scaled)

    def mouseDoubleClickEvent(self, event):
        self._close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Space):
            self._close()
        super().keyPressEvent(event)

    def _close(self):
        self.hide()
        self.closed.emit()


class CameraLabel(QLabel):
    """
    Camera display label.
    Double-click (desktop) or double-tap (touch) expands the feed to full screen.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self._last_click_ms = 0
        self._last_pixmap   = None
        self._fs_win        = FullScreenCameraWindow()
        self._fs_win.closed.connect(self._on_fs_closed)
        self._fs_open       = False

        # Hint overlay text
        self._hint_lbl = QLabel("⤢  Double-tap to expand", self)
        self._hint_lbl.setStyleSheet(
            f"background:rgba(5,13,7,180);color:{C['muted']};"
            f"font-size:9px;letter-spacing:1px;padding:3px 8px;"
            f"border-radius:4px;")
        self._hint_lbl.adjustSize()
        self._hint_lbl.move(6, 6)
        self._hint_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_camera_pixmap(self, pix: QPixmap):
        """Call this instead of setPixmap so the full-screen window stays in sync."""
        self._last_pixmap = pix
        self.setPixmap(pix)
        if self._fs_open:
            self._fs_win.update_frame(pix)

    # ── Event handlers ──────────────────────────────────────────────────────
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._open_fullscreen()

    def mousePressEvent(self, event):
        """Fallback double-tap detection via rapid single clicks (touch screens)."""
        if event.button() == Qt.LeftButton:
            now = time.time() * 1000
            if now - self._last_click_ms < 400:
                self._open_fullscreen()
                self._last_click_ms = 0
            else:
                self._last_click_ms = now
        super().mousePressEvent(event)

    def event(self, ev):
        """Native touch: detect two taps within 400 ms."""
        from PyQt5.QtCore import QEvent
        if ev.type() == QEvent.TouchBegin or ev.type() == QEvent.TouchUpdate:
            ev.accept()
            return True
        if ev.type() == QEvent.TouchEnd:
            now = time.time() * 1000
            if now - self._last_click_ms < 400:
                self._open_fullscreen()
                self._last_click_ms = 0
            else:
                self._last_click_ms = now
            ev.accept()
            return True
        return super().event(ev)

    # ── Fullscreen toggle ────────────────────────────────────────────────────
    def _open_fullscreen(self):
        if self._fs_open:
            return
        self._fs_open = True
        if self._last_pixmap:
            self._fs_win.update_frame(self._last_pixmap)
        self._fs_win.show_fullscreen()

    def _on_fs_closed(self):
        self._fs_open = False


class Card(QFrame):
    def __init__(self, p=None):
        super().__init__(p); self.setObjectName("card")
        QVBoxLayout(self).setContentsMargins(0,0,0,0)
        self.layout().setSpacing(0)
    def add(self, w): self.layout().addWidget(w)

class SecHdr(QLabel):
    def __init__(self, txt, p=None):
        super().__init__(txt.upper(), p)
        self.setStyleSheet(
            f"background:{C['bg2']};border-bottom:1px solid {C['bdr2']};"
            f"border-radius:12px 12px 0 0;color:{C['grn3']};font-size:11px;"
            f"letter-spacing:2px;font-weight:bold;padding:10px 14px;")

class MBox(QFrame):
    def __init__(self, label, val="—", p=None):
        super().__init__(p); self.setObjectName("card_inner")
        lay = QVBoxLayout(self); lay.setSpacing(3); lay.setContentsMargins(12,10,12,10)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;letter-spacing:2px;")
        self.v = QLabel(val)
        self.v.setStyleSheet(f"color:{C['grn3']};font-size:18px;font-weight:bold;")
        lay.addWidget(lbl); lay.addWidget(self.v)
    def set(self, val, col=None):
        self.v.setText(str(val))
        if col: self.v.setStyleSheet(f"color:{col};font-size:18px;font-weight:bold;")

class SigBar(QWidget):
    def __init__(self, label, p=None):
        super().__init__(p)
        lay = QHBoxLayout(self); lay.setContentsMargins(14,4,14,4); lay.setSpacing(10)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;letter-spacing:1px;")
        lbl.setFixedWidth(50)
        self.bar = QProgressBar(); self.bar.setRange(0,100); self.bar.setFixedHeight(6)
        self.vl  = QLabel("—")
        self.vl.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        self.vl.setFixedWidth(120); self.vl.setAlignment(Qt.AlignRight)
        lay.addWidget(lbl); lay.addWidget(self.bar); lay.addWidget(self.vl)
    def update(self, label: str, conf: float):
        self.bar.setValue(int(conf * 100))
        self.vl.setText(f"{label}  {conf:.2f}")
        self.bar.setObjectName("bar_green" if label == "Focused" else "bar_red")
        self.bar.setStyle(self.bar.style())

class RadarWidget(QWidget):
    def __init__(self, p=None):
        super().__init__(p); self.setMinimumSize(130,130)
        self.yaw = self.pitch = self.roll = 0.
    def set_pose(self, y, pi, r): self.yaw=y; self.pitch=pi; self.roll=r; self.update()
    def paintEvent(self, _):
        p  = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        cx = self.width()//2; cy = self.height()//2
        r  = min(self.width(), self.height())//2 - 10
        for sc in (.33, .66, 1.):
            p.setPen(QPen(QColor(C['bdr2']),1))
            p.drawEllipse(cx-int(r*sc), cy-int(r*sc), int(r*sc*2), int(r*sc*2))
        axes = [(0,-r),(int(r*.87),int(r*.5)),(-int(r*.87),int(r*.5))]
        for ax, ay in axes:
            p.setPen(QPen(QColor(C['bdr2']),1)); p.drawLine(cx,cy,cx+ax,cy+ay)
        def c01(v): return max(0., min(1., v))
        yn=c01((self.yaw+.4)/.8); pn=c01((self.pitch+.3)/.6); rn=c01((self.roll+.2)/.4)
        pts = [QPoint(cx+int(axes[0][0]*yn),cy+int(axes[0][1]*yn)),
               QPoint(cx+int(axes[1][0]*pn),cy+int(axes[1][1]*pn)),
               QPoint(cx+int(axes[2][0]*rn),cy+int(axes[2][1]*rn))]
        fill = QColor(C['grn1']); fill.setAlpha(60)
        p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(C['grn2']),1.5))
        p.drawPolygon(QPolygon(pts))
        for pt in pts:
            p.setBrush(QBrush(QColor(C['grn3']))); p.setPen(Qt.NoPen)
            p.drawEllipse(pt, 4, 4)
        p.setPen(QPen(QColor(C['muted']))); p.setFont(QFont("Courier New",8))
        p.drawText(cx-10, cy-r-4, "YAW")
        p.drawText(cx+int(r*.87)+4,  cy+int(r*.5)+14, "PCH")
        p.drawText(cx-int(r*.87)-28, cy+int(r*.5)+14, "ROL")

class DistGauge(QWidget):
    def __init__(self, p=None):
        super().__init__(p); self.setMinimumSize(120,120); self.val = 0.
    def set(self, v): self.val = v; self.update()
    def paintEvent(self, _):
        p  = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W  = self.width(); H = self.height(); cx=W//2; cy=H//2
        r  = min(W, H)//2 - 12
        p.setPen(QPen(QColor(C['bg3']),8,Qt.SolidLine,Qt.RoundCap))
        p.drawArc(QRectF(cx-r,cy-r,r*2,r*2), 225*16, -270*16)
        frac = min(self.val/MAX_DISTANCE_CM, 1.); span = int(-270*frac)*16
        col  = QColor(C['red']) if self.val>90 else (QColor(C['yel']) if self.val<30 else QColor(C['grn1']))
        p.setPen(QPen(col,8,Qt.SolidLine,Qt.RoundCap))
        p.drawArc(QRectF(cx-r,cy-r,r*2,r*2), 225*16, span)
        p.setPen(QPen(QColor(C['grn4']))); p.setFont(QFont("Courier New",16,QFont.Bold))
        p.drawText(QRectF(cx-r,cy-12,r*2,24), Qt.AlignCenter, f"{self.val:.0f}")
        p.setFont(QFont("Courier New",8)); p.setPen(QPen(QColor(C['muted'])))
        p.drawText(QRectF(cx-r,cy+10,r*2,20), Qt.AlignCenter, "cm")


# ══════════════════════════════════════════════════════════════════════════════
#  FocusSenseWindow  —  Main application window
# ══════════════════════════════════════════════════════════════════════════════
class FocusSenseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"FocusSense  —  AI Attention Tracker  [⚠ every {WARNING_INTERVAL_S}s]")
        self.setMinimumSize(1340, 820)
        self.setStyleSheet(QSS)

        # Session counters
        self._session_secs  = 0
        self._total_frames  = 0
        self._focus_frames  = 0
        self._distract_cnt  = 0
        self._dist_min      = float("inf")
        self._dist_max      = 0.
        self._prev_focused  = None
        self._warn_count    = 0
        self._app_was_active = True

        self._build_ui()
        self._build_tray()
        self._start_workers()
        self._start_timers()

    # ──────────────────────────────────────────────────────────────────────────
    #  System tray
    # ──────────────────────────────────────────────────────────────────────────
    def _build_tray(self):
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(
            self.style().standardIcon(self.style().SP_MessageBoxWarning))
        self._tray.setToolTip("FocusSense — Attention Tracker")
        menu = QMenu()
        menu.addAction(QAction("Show Window", self, triggered=self.show))
        menu.addAction(QAction("Quit",        self, triggered=QApplication.quit))
        self._tray.setContextMenu(menu)
        self._tray.show()

    def _tray_notify(self, title: str, msg: str):
        if self._tray.isSystemTrayAvailable():
            self._tray.showMessage(title, msg, QSystemTrayIcon.Critical, 7000)

    # ──────────────────────────────────────────────────────────────────────────
    #  Timers
    # ──────────────────────────────────────────────────────────────────────────
    def _start_timers(self):
        # Clock + session timer
        QTimer(self, interval=1000, timeout=self._tick_clock).start()
        # App-focus watcher
        QTimer(self, interval=2000, timeout=self._check_app_focus).start()

    def _tick_clock(self):
        t = QTime.currentTime().toString("hh:mm:ss")
        self._clock_lbl.setText(t)
        self._ft_ts.setText(f"Last update: {t}")
        self._session_secs += 1
        m = str(self._session_secs//60).zfill(2)
        s = str(self._session_secs%60).zfill(2)
        self._m_session.set(f"{m}:{s}")
        self._dot.setText("●" if self._session_secs % 2 == 0 else "○")

    def _check_app_focus(self):
        """Send OS notification if the user switched to another app."""
        active = QApplication.activeWindow() is not None
        if self._app_was_active and not active:
            self._log_msg("User switched to another application.", "warn")
            self._tray_notify(
                "FocusSense — Stay Productive! 🎯",
                "You left your work.\n"
                "Close distracting apps and stay focused!"
            )
        self._app_was_active = active

    # ──────────────────────────────────────────────────────────────────────────
    #  Workers
    # ──────────────────────────────────────────────────────────────────────────
    def _start_workers(self):
        # Camera + ML worker (imports AttentionEngine from main.py)
        self._worker = InferenceWorker()
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_update.connect(lambda m: self._log_msg(m, "info"))
        self._worker.start()

        # Distraction timer
        self._warn_engine = DistractionWarningEngine()
        self._warn_engine.warn_fire.connect(self._fire_warning)
        self._warn_engine.tick_update.connect(self._on_warn_tick)
        self._warn_engine.start()

    # ──────────────────────────────────────────────────────────────────────────
    #  Build UI
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        ml = QVBoxLayout(root); ml.setContentsMargins(18,14,18,14); ml.setSpacing(14)
        ml.addWidget(self._make_header())
        ml.addWidget(self._make_top_metrics())
        body = QHBoxLayout(); body.setSpacing(14)
        body.addWidget(self._make_left(),  0)
        body.addWidget(self._make_right(), 1)
        ml.addLayout(body, 1)
        ml.addWidget(self._make_footer())

        # Overlays
        self._popup      = WarningPopup(root)
        self._fullscreen = FullScreenRedFlash()

    # ── Header ──────────────────────────────────────────────────────────────
    def _make_header(self):
        w   = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        logo = QWidget(); ll = QHBoxLayout(logo); ll.setSpacing(12); ll.setContentsMargins(0,0,0,0)
        icon = QLabel("◎")
        icon.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f"stop:0 {C['grn6']},stop:1 {C['grn1']});"
            f"color:{C['grn3']};font-size:22px;border-radius:10px;padding:7px 10px;")
        tc = QWidget(); tl = QVBoxLayout(tc); tl.setSpacing(2); tl.setContentsMargins(0,0,0,0)
        h1 = QLabel("FocusSense")
        h1.setStyleSheet(f"color:{C['grn4']};font-size:20px;font-weight:bold;letter-spacing:-1px;")
        h2 = QLabel(
            f"AI ATTENTION TRACKER  v3.0"
            f"  ·  ⚠ WARN EVERY {WARNING_INTERVAL_S}s"
            f"  ·  EYE GAZE TRACKING"
            f"  ·  Powered by main.py + Test_1.py")
        h2.setStyleSheet(f"color:{C['muted']};font-size:9px;letter-spacing:2px;")
        tl.addWidget(h1); tl.addWidget(h2); ll.addWidget(icon); ll.addWidget(tc)
        rw = QWidget(); rl = QHBoxLayout(rw); rl.setSpacing(16); rl.setContentsMargins(0,0,0,0)
        self._clock_lbl = QLabel("--:--:--")
        self._clock_lbl.setStyleSheet(f"color:{C['txt2']};font-size:13px;letter-spacing:2px;")
        badge = QWidget()
        badge.setStyleSheet(
            f"background:{C['bg2']};border:1px solid {C['bdr2']};border-radius:8px;")
        bl = QHBoxLayout(badge); bl.setSpacing(8); bl.setContentsMargins(12,7,12,7)
        self._dot = QLabel("●"); self._dot.setStyleSheet(f"color:{C['grn2']};font-size:10px;")
        self._sys_lbl = QLabel("SYSTEM ONLINE")
        self._sys_lbl.setStyleSheet(f"color:{C['txt']};font-size:11px;letter-spacing:1px;")
        bl.addWidget(self._dot); bl.addWidget(self._sys_lbl)
        rl.addWidget(self._clock_lbl); rl.addWidget(badge)
        lay.addWidget(logo); lay.addStretch(); lay.addWidget(rw)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{C['bdr2']};margin-top:4px;")
        c = QWidget(); cl = QVBoxLayout(c); cl.setContentsMargins(0,0,0,0); cl.setSpacing(8)
        cl.addWidget(w); cl.addWidget(sep)
        return c

    # ── Top metrics ─────────────────────────────────────────────────────────
    def _make_top_metrics(self):
        w = QWidget(); lay = QHBoxLayout(w); lay.setSpacing(12); lay.setContentsMargins(0,0,0,0)
        self._m_focus   = MBox("Focus Rate",      "—")
        self._m_events  = MBox("Distract Events", "0")
        self._m_warns   = MBox("Warnings Fired",  "0")
        self._m_gaze    = MBox("Gaze Zone",        "—")
        self._m_session = MBox("Session Time",    "00:00")
        for m in [self._m_focus, self._m_events, self._m_warns,
                  self._m_gaze, self._m_session]:
            lay.addWidget(m)
        return w

    # ── Left panel ──────────────────────────────────────────────────────────
    def _make_left(self):
        w = QWidget(); w.setFixedWidth(355)
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)

        # Camera feed
        cc = Card(); cc.add(SecHdr("Camera Feed"))
        cb = QWidget(); cbl = QVBoxLayout(cb); cbl.setContentsMargins(12,12,12,12)
        self._cam_lbl = CameraLabel()
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setMinimumHeight(200)
        self._cam_lbl.setStyleSheet(
            f"background:{C['bg3']};border:1px solid {C['bdr2']};"
            f"border-radius:8px;color:{C['muted']};font-size:12px;")
        self._cam_lbl.setText("⬤  INITIALIZING…")
        self._cam_lbl.setCursor(Qt.PointingHandCursor)
        cbl.addWidget(self._cam_lbl); cc.add(cb)

        # Live status
        sc = Card(); sc.add(SecHdr("Live Status"))
        sb = QWidget(); sl = QVBoxLayout(sb); sl.setContentsMargins(14,12,14,12); sl.setSpacing(8)
        self._status_lbl = QLabel("WAITING…")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet(
            f"font-size:24px;font-weight:bold;color:{C['grn3']};letter-spacing:3px;")
        cl_lbl = QLabel("CONFIDENCE")
        cl_lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;letter-spacing:2px;")
        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0,100); self._conf_bar.setFixedHeight(6)
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setStyleSheet(f"color:{C['bdr']};")
        self._sig_gaze = SigBar("Gaze")
        self._sig_pose = SigBar("Pose")
        self._sig_emo  = SigBar("Face")
        sl.addWidget(self._status_lbl); sl.addWidget(cl_lbl)
        sl.addWidget(self._conf_bar); sl.addWidget(sep2)
        sl.addWidget(self._sig_gaze); sl.addWidget(self._sig_pose); sl.addWidget(self._sig_emo)
        sc.add(sb)

        # Warning countdown
        wc = Card(); wc.add(SecHdr(f"⚠  Warning Timer  (fires @{WARNING_INTERVAL_S}s)"))
        wb = QWidget(); wl = QVBoxLayout(wb); wl.setContentsMargins(14,12,14,14); wl.setSpacing(8)
        row1 = QHBoxLayout()
        self._warn_secs_lbl = QLabel("0 s")
        self._warn_secs_lbl.setStyleSheet(f"color:{C['grn2']};font-size:24px;font-weight:bold;")
        sc2 = QVBoxLayout(); sc2.setSpacing(2)
        self._warn_state_lbl = QLabel("FOCUSED — timer paused")
        self._warn_state_lbl.setStyleSheet(f"color:{C['grn2']};font-size:11px;")
        self._warn_next_lbl  = QLabel(f"Next warning at {WARNING_INTERVAL_S}s")
        self._warn_next_lbl.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        sc2.addWidget(self._warn_state_lbl); sc2.addWidget(self._warn_next_lbl)
        row1.addWidget(self._warn_secs_lbl); row1.addLayout(sc2); row1.addStretch()
        pb_lbl = QLabel("PROGRESS TO NEXT WARNING")
        pb_lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;letter-spacing:1px;")
        self._warn_cd_bar = QProgressBar()
        self._warn_cd_bar.setRange(0,100); self._warn_cd_bar.setValue(0)
        self._warn_cd_bar.setFixedHeight(8); self._warn_cd_bar.setObjectName("bar_yel")
        self._warn_cd_bar.setStyle(self._warn_cd_bar.style())
        info = QLabel(
            f"Warning fires every {WARNING_INTERVAL_S}s of continuous distraction.\n"
            "Resets the instant focus is restored.")
        info.setStyleSheet(f"color:{C['muted']};font-size:10px;"); info.setWordWrap(True)
        self._total_warn_lbl = QLabel("Total warnings this session:  0")
        self._total_warn_lbl.setStyleSheet(
            f"background:{C['bg2']};border:1px solid {C['bdr2']};border-radius:6px;"
            f"color:{C['grn3']};font-size:11px;padding:6px 10px;")
        wl.addLayout(row1); wl.addWidget(pb_lbl); wl.addWidget(self._warn_cd_bar)
        wl.addWidget(info); wl.addWidget(self._total_warn_lbl); wc.add(wb)

        # Control buttons
        bw = QWidget(); bl2 = QHBoxLayout(bw); bl2.setSpacing(8); bl2.setContentsMargins(0,0,0,0)
        self._btn_toggle = QPushButton("▶  START"); self._btn_toggle.setObjectName("btn_primary")
        self._btn_toggle.clicked.connect(self._toggle_cam)
        btn_exp = QPushButton("↓  EXPORT"); btn_exp.clicked.connect(self._export_log)
        btn_stp = QPushButton("■  STOP");   btn_stp.setObjectName("btn_danger")
        btn_stp.clicked.connect(self._stop_session)
        bl2.addWidget(self._btn_toggle); bl2.addWidget(btn_exp); bl2.addWidget(btn_stp)

        lay.addWidget(cc); lay.addWidget(sc); lay.addWidget(wc); lay.addWidget(bw)
        return w

    # ── Right panel ─────────────────────────────────────────────────────────
    def _make_right(self):
        w   = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)

        # Gauges row
        gauges = QHBoxLayout(); gauges.setSpacing(12)

        # Eye gaze screen widget
        gc = Card(); gc.add(SecHdr("👁  Eye Gaze Position"))
        gb = QWidget(); gl = QVBoxLayout(gb); gl.setContentsMargins(12,10,12,10)
        self._gaze_screen = GazeScreenWidget()
        self._gaze_screen.setMinimumHeight(168)
        coord_row = QHBoxLayout()
        self._gaze_x   = MBox("Gaze X",  "0.50")
        self._gaze_y   = MBox("Gaze Y",  "0.50")
        self._gaze_zone = MBox("Zone",   "CENTER")
        for m in [self._gaze_x, self._gaze_y, self._gaze_zone]:
            m.v.setStyleSheet(f"color:{C['grn3']};font-size:14px;font-weight:bold;")
            coord_row.addWidget(m)
        gl.addWidget(self._gaze_screen); gl.addLayout(coord_row); gc.add(gb)

        # Distance gauge
        dc = Card(); dc.add(SecHdr("Distance"))
        db = QWidget(); dl = QVBoxLayout(db); dl.setContentsMargins(12,8,12,12); dl.setAlignment(Qt.AlignCenter)
        self._gauge   = DistGauge()
        self._dist_st = QLabel("NO FACE"); self._dist_st.setAlignment(Qt.AlignCenter)
        self._dist_st.setStyleSheet(f"color:{C['muted']};font-size:9px;letter-spacing:2px;")
        ds = QHBoxLayout()
        self._dmin = MBox("Min","—"); self._dmax = MBox("Max","—")
        for m in [self._dmin, self._dmax]:
            m.v.setStyleSheet(f"color:{C['grn3']};font-size:13px;font-weight:bold;")
            ds.addWidget(m)
        dl.addWidget(self._gauge); dl.addWidget(self._dist_st); dl.addLayout(ds); dc.add(db)

        # Head pose radar
        pc = Card(); pc.add(SecHdr("Head Pose"))
        pb = QWidget(); pl2 = QVBoxLayout(pb); pl2.setContentsMargins(12,8,12,12); pl2.setAlignment(Qt.AlignCenter)
        self._radar = RadarWidget()
        pvl = QHBoxLayout()
        self._yaw_lbl = MBox("Yaw","0.00"); self._pit_lbl = MBox("Pitch","0.00"); self._rol_lbl = MBox("Roll","0.00")
        for m in [self._yaw_lbl, self._pit_lbl, self._rol_lbl]:
            m.v.setStyleSheet(f"color:{C['grn3']};font-size:13px;font-weight:bold;")
            pvl.addWidget(m)
        pl2.addWidget(self._radar); pl2.addLayout(pvl); pc.add(pb)

        gauges.addWidget(gc, 2); gauges.addWidget(dc, 1); gauges.addWidget(pc, 1)

        # Tabs
        tabs = QTabWidget(); tabs.setStyleSheet(QSS)

        # Event log tab
        lt = QWidget(); ll = QVBoxLayout(lt); ll.setContentsMargins(8,8,8,8)
        self._log = QTextEdit(); self._log.setReadOnly(True); ll.addWidget(self._log)
        tabs.addTab(lt, "EVENT LOG")

        # Warnings history tab
        wt  = QWidget(); wl2 = QVBoxLayout(wt); wl2.setContentsMargins(8,8,8,8)
        self._warn_log = QTextEdit(); self._warn_log.setReadOnly(True)
        self._warn_log.append(
            f'<span style="color:{C["muted"]}">Warning history will appear here. '
            f'Interval: {WARNING_INTERVAL_S}s.</span>')
        wl2.addWidget(self._warn_log)
        tabs.addTab(wt, "⚠  WARNINGS")

        # Architecture info tab
        at  = QWidget(); al = QVBoxLayout(at); al.setContentsMargins(12,12,12,12)
        arch_txt = QTextEdit(); arch_txt.setReadOnly(True)
        arch_txt.append(
            f'<b style="color:{C["grn3"]}">3-FILE ARCHITECTURE</b><br><br>'
            f'<span style="color:{C["txt2"]}">'
            f'<b style="color:{C["grn2"]}">Test_1.py</b>  —  Model Training<br>'
            f'&nbsp;&nbsp;• Downloads RAVDESS + AFLW2000-3D datasets<br>'
            f'&nbsp;&nbsp;• Trains gaze_model.keras  (Dense 128→64→32)<br>'
            f'&nbsp;&nbsp;• Trains pose_model.keras  (Dense 64→32→16)<br>'
            f'&nbsp;&nbsp;• Saves .keras + .pkl artefacts<br><br>'
            f'<b style="color:{C["grn2"]}">main.py</b>  —  Inference Backend<br>'
            f'&nbsp;&nbsp;• AttentionEngine class  (importable)<br>'
            f'&nbsp;&nbsp;• estimate_distance(), get_gaze_vector()<br>'
            f'&nbsp;&nbsp;• get_head_pose_vector(), estimate_gaze_screen_point()<br>'
            f'&nbsp;&nbsp;• predict_gaze(), predict_pose(), deepface_attention()<br>'
            f'&nbsp;&nbsp;• Standalone: python main.py  (raw OpenCV window)<br><br>'
            f'<b style="color:{C["grn2"]}">Front_1.py</b>  —  Qt Frontend  (THIS FILE)<br>'
            f'&nbsp;&nbsp;• Imports AttentionEngine from main.py<br>'
            f'&nbsp;&nbsp;• InferenceWorker QThread runs engine.process()<br>'
            f'&nbsp;&nbsp;• DistractionWarningEngine fires every {WARNING_INTERVAL_S}s<br>'
            f'&nbsp;&nbsp;• Full-screen red flash + popup + OS tray notification<br>'
            f'&nbsp;&nbsp;• Eye gaze position shown on screen preview widget<br>'
            f'</span>'
        )
        al.addWidget(arch_txt)
        tabs.addTab(at, "ARCHITECTURE")

        lay.addLayout(gauges); lay.addWidget(tabs, 1)
        return w

    def _make_footer(self):
        w   = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,6,0,0)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{C['bdr2']};margin-top:4px;")
        self._ft_lbl = QLabel(
            "FocusSense  ·  Front_1.py  →  main.py (AttentionEngine)  →  Test_1.py (models)")
        self._ft_lbl.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        self._ft_ts  = QLabel("—"); self._ft_ts.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        lay.addWidget(self._ft_lbl); lay.addStretch(); lay.addWidget(self._ft_ts)
        c = QWidget(); cl = QVBoxLayout(c); cl.setContentsMargins(0,0,0,0); cl.setSpacing(4)
        cl.addWidget(sep); cl.addWidget(w)
        return c

    # ──────────────────────────────────────────────────────────────────────────
    #  Frame update  —  called ~30 fps from InferenceWorker
    # ──────────────────────────────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray, state: dict):
        # Camera display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pix = QPixmap.fromImage(
            QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        ).scaled(self._cam_lbl.width(), self._cam_lbl.height(),
                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cam_lbl.set_camera_pixmap(pix)

        focused = state.get("focused", False)
        self._total_frames += 1
        if focused: self._focus_frames += 1

        # Feed the 15-second warning engine
        self._warn_engine.set_focused(focused)

        # Status label
        scol = C["grn2"] if focused else C["red"]
        self._status_lbl.setText("FOCUSED" if focused else "DISTRACTED")
        self._status_lbl.setStyleSheet(
            f"font-size:24px;font-weight:bold;color:{scol};letter-spacing:3px;")

        gc = state.get("gaze_conf", 0.); pc = state.get("pose_conf", 0.)
        conf = gc if focused else 1 - gc
        self._conf_bar.setValue(int(conf * 100))
        self._conf_bar.setObjectName("bar_green" if focused else "bar_red")
        self._conf_bar.setStyle(self._conf_bar.style())

        self._sig_gaze.update(state.get("gaze_label", "—"), gc)
        self._sig_pose.update(state.get("pose_label", "—"), pc)
        self._sig_emo.update(state.get("df_label", "—"), 0.6)

        # Gaze position
        gx = state.get("gaze_nx", 0.5); gy = state.get("gaze_ny", 0.5)
        self._gaze_screen.set_gaze(gx, gy, focused)
        self._gaze_x.set(f"{gx:.2f}")
        self._gaze_y.set(f"{gy:.2f}")
        hz   = "LEFT" if gx < .33 else ("RIGHT" if gx > .67 else "CENTER")
        vt   = "TOP"  if gy < .33 else ("BOTTOM" if gy > .67 else "")
        zone = f"{vt} {hz}".strip() if vt else hz
        self._gaze_zone.set(zone, C["grn2"] if focused else C["red"])
        self._m_gaze.set(zone, C["grn2"] if focused else C["red"])

        # Distance
        d = state.get("dist", 0.)
        if 0 < d < 500:
            self._gauge.set(d)
            self._dist_st.setText(
                "TOO FAR"   if d > 90 else
                "TOO CLOSE" if d < 30 else "OPTIMAL RANGE")
            if d < self._dist_min: self._dist_min = d; self._dmin.set(f"{d:.0f}cm")
            if d > self._dist_max: self._dist_max = d; self._dmax.set(f"{d:.0f}cm")

        # Head pose
        yaw = state.get("yaw", 0.); pit = state.get("pitch", 0.); rol = state.get("roll", 0.)
        self._radar.set_pose(yaw, pit, rol)
        self._yaw_lbl.set(f"{yaw:.2f}")
        self._pit_lbl.set(f"{pit:.2f}")
        self._rol_lbl.set(f"{rol:.2f}")

        # Top metrics
        if self._total_frames > 0:
            rate = int(self._focus_frames / self._total_frames * 100)
            self._m_focus.set(f"{rate}%", C["grn2"] if rate >= 70 else C["yel"])
        self._m_events.set(self._distract_cnt)

        # Log state changes
        if self._prev_focused is not None and focused != self._prev_focused:
            if focused:
                self._log_msg("Gaze/Pose → FOCUSED  (distraction timer reset)", "ok")
            else:
                self._distract_cnt += 1
                self._log_msg("Gaze/Pose → DISTRACTED  (15s timer started)", "warn")
        self._prev_focused = focused

    # ──────────────────────────────────────────────────────────────────────────
    #  Warning engine callbacks
    # ──────────────────────────────────────────────────────────────────────────
    def _on_warn_tick(self, distract_secs: int, pct: int):
        focused = distract_secs == 0
        if focused:
            self._warn_secs_lbl.setText("0 s")
            self._warn_secs_lbl.setStyleSheet(f"color:{C['grn2']};font-size:24px;font-weight:bold;")
            self._warn_state_lbl.setText("FOCUSED — timer paused")
            self._warn_state_lbl.setStyleSheet(f"color:{C['grn2']};font-size:11px;")
            self._warn_next_lbl.setText(f"Next warning at {WARNING_INTERVAL_S}s")
            self._warn_cd_bar.setValue(0); self._warn_cd_bar.setObjectName("bar_yel")
        else:
            remaining = WARNING_INTERVAL_S - (distract_secs % WARNING_INTERVAL_S)
            col     = C["yel"] if pct < 40 else ("#f97316" if pct < 75 else C["red"])
            bar_obj = "bar_yel" if pct < 40 else ("bar_orange" if pct < 75 else "bar_red")
            self._warn_secs_lbl.setText(f"{distract_secs} s")
            self._warn_secs_lbl.setStyleSheet(f"color:{col};font-size:24px;font-weight:bold;")
            self._warn_state_lbl.setText(
                f"DISTRACTED  ·  {distract_secs}s  ·  next warning in {remaining}s")
            self._warn_state_lbl.setStyleSheet(f"color:{col};font-size:11px;")
            self._warn_next_lbl.setText(
                f"Interval: {WARNING_INTERVAL_S}s  ·  Fired: {self._warn_count}")
            self._warn_cd_bar.setValue(pct); self._warn_cd_bar.setObjectName(bar_obj)
        self._warn_cd_bar.setStyle(self._warn_cd_bar.style())

    def _fire_warning(self, warn_n: int, distract_secs: int):
        self._warn_count = warn_n
        self._m_warns.set(str(warn_n), C["red"])
        self._total_warn_lbl.setText(f"Total warnings this session:  {warn_n}")

        # 1 — Full-screen red flash
        self._fullscreen.flash(warn_n, distract_secs)

        # 2 — Slide-in popup inside app
        self._popup.show_warning(warn_n, distract_secs)

        # 3 — OS tray notification (visible even in other apps)
        self._tray_notify(
            f"⚠ FocusSense — Warning #{warn_n}",
            f"Distracted for {distract_secs}s.\n"
            "Close distracting apps and refocus!"
        )

        # 4 — Audio beep
        QApplication.beep()

        # 5 — Both logs
        ts = QTime.currentTime().toString("hh:mm:ss")
        self._log_msg(f"⚠ WARNING #{warn_n} — {distract_secs}s of continuous distraction", "err")
        self._warn_log.append(
            f'<b style="color:{C["red"]}">⚠ WARNING #{warn_n}</b>'
            f'  <span style="color:{C["muted"]}">[{ts}]</span><br>'
            f'<span style="color:{C["yel"]}">Distracted for {distract_secs}s.</span><br>'
            f'<span style="color:{C["muted"]}">OS notification + full-screen flash triggered.</span>'
            f'<hr color="{C["bdr2"]}">'
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Controls
    # ──────────────────────────────────────────────────────────────────────────
    def _log_msg(self, msg: str, level: str = "info"):
        ts   = QTime.currentTime().toString("hh:mm:ss")
        cols = {"ok":C["grn2"],"warn":C["yel"],"err":C["red"],"info":C["grn3"]}
        col  = cols.get(level, C["txt2"])
        self._log.append(
            f'<span style="color:{C["muted"]}">[{ts}]</span> '
            f'<span style="color:{col}">{msg}</span>')

    def _toggle_cam(self):
        self._worker.toggle_pause()
        paused = self._worker._paused
        self._btn_toggle.setText("▶  RESUME" if paused else "⏸  PAUSE")
        self._log_msg("Camera paused." if paused else "Camera resumed.", "warn")

    def _export_log(self):
        text = self._log.toPlainText()
        path = f"focussense_log_{int(time.time())}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._log_msg(f"Log exported → {path}", "ok")

    def _stop_session(self):
        self._worker.stop(); self._warn_engine.stop()
        self._dot.setText("●")
        self._dot.setStyleSheet(f"color:{C['red']};font-size:10px;")
        self._sys_lbl.setText("SESSION STOPPED")
        self._log_msg("Session stopped by user.", "err")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_popup"):
            pass   # popup positions itself relative to parent on show

    def closeEvent(self, event):
        self._worker.stop(); self._warn_engine.stop()
        self._worker.wait(2000); self._warn_engine.wait(2000)
        self._tray.hide()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("FocusSense")
    app.setApplicationDisplayName("FocusSense")
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(C["bg0"]))
    pal.setColor(QPalette.WindowText,      QColor(C["txt"]))
    pal.setColor(QPalette.Base,            QColor(C["bg1"]))
    pal.setColor(QPalette.AlternateBase,   QColor(C["bg2"]))
    pal.setColor(QPalette.Text,            QColor(C["txt"]))
    pal.setColor(QPalette.Button,          QColor(C["bg2"]))
    pal.setColor(QPalette.ButtonText,      QColor(C["txt"]))
    pal.setColor(QPalette.Highlight,       QColor(C["grn1"]))
    pal.setColor(QPalette.HighlightedText, QColor(C["grn4"]))
    app.setPalette(pal)

    win = FocusSenseWindow()
    win.show()
    sys.exit(app.exec_())


# """
# Front_1.py — NeuralEye Attention Tracker (Main Application)
# ============================================================
# Run this file to launch the application.

# Features:
#   • Full dashboard UI matching the HTML design
#   • Real-time gaze tracking with screen-point visualization
#   • Theme switcher (Dark Purple / Cyberpunk Green / Ocean Blue / Light)
#   • Background-detection enforcement: if the app is backgrounded,
#     all other user apps are closed and a 15-second countdown is shown
#   • Imports AttentionEngine from main.py for ML inference

# Requirements:
#   pip install PyQt6 opencv-python mediapipe deepface tensorflow
#               joblib scikit-learn scipy numpy certifi pygetwindow psutil

# Run:
#   python Front_1.py
# """

# import sys
# import os
# import time
# import threading
# import platform
# import subprocess
# import math

# import cv2
# import numpy as np

# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QGridLayout, QLabel, QPushButton, QFrame, QTabWidget, QTextEdit,
#     QProgressBar, QSizePolicy, QComboBox, QDialog, QGraphicsOpacityEffect,
#     QScrollArea, QSpacerItem
# )
# from PyQt6.QtCore import (
#     Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QSize,
#     QPropertyAnimation, QEasingCurve
# )
# from PyQt6.QtGui import (
#     QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont,
#     QFontDatabase, QLinearGradient, QPalette, QIcon, QConicalGradient
# )

# # ── Import the attention engine ───────────────────────────────────────────────
# try:
#     from main import AttentionEngine, WARNING_INTERVAL_S
#     ENGINE_AVAILABLE = True
# except ImportError as _e:
#     print(f"[WARN] main.py not importable: {_e}. Demo mode active.")
#     ENGINE_AVAILABLE = False
#     WARNING_INTERVAL_S = 15


# # ══════════════════════════════════════════════════════════════════════════════
# #  THEMES
# # ══════════════════════════════════════════════════════════════════════════════
# THEMES = {
#     "Dark Purple": {
#         "bg0":    "#07060d",
#         "bg1":    "#0e0c1a",
#         "bg2":    "#14112a",
#         "bg3":    "#1c1836",
#         "accent": "#7c3aed",
#         "accent2":"#9d5cf6",
#         "accent3":"#c084fc",
#         "accent4":"#e9d5ff",
#         "accent5":"#4c1d95",
#         "green":  "#22c55e",
#         "red":    "#ef4444",
#         "yellow": "#f59e0b",
#         "cyan":   "#06b6d4",
#         "text":   "#e2d9f3",
#         "text2":  "#a78bca",
#         "text3":  "#6b4d8c",
#         "border": "rgba(124,58,237,0.25)",
#     },
#     "Cyberpunk Green": {
#         "bg0":    "#020c05",
#         "bg1":    "#050f08",
#         "bg2":    "#0a1f10",
#         "bg3":    "#0f2a18",
#         "accent": "#00ff88",
#         "accent2":"#00cc6a",
#         "accent3":"#00ff44",
#         "accent4":"#ccffe0",
#         "accent5":"#003d20",
#         "green":  "#00ff88",
#         "red":    "#ff3355",
#         "yellow": "#ffdd00",
#         "cyan":   "#00ffcc",
#         "text":   "#d0ffe8",
#         "text2":  "#66cc88",
#         "text3":  "#336644",
#         "border": "rgba(0,255,136,0.25)",
#     },
#     "Ocean Blue": {
#         "bg0":    "#020810",
#         "bg1":    "#040e1c",
#         "bg2":    "#071525",
#         "bg3":    "#0b1e35",
#         "accent": "#0ea5e9",
#         "accent2":"#38bdf8",
#         "accent3":"#7dd3fc",
#         "accent4":"#e0f2fe",
#         "accent5":"#0c4a6e",
#         "green":  "#22c55e",
#         "red":    "#ef4444",
#         "yellow": "#f59e0b",
#         "cyan":   "#06b6d4",
#         "text":   "#e0f2fe",
#         "text2":  "#7dd3fc",
#         "text3":  "#1e6a9e",
#         "border": "rgba(14,165,233,0.25)",
#     },
#     "Light Mode": {
#         "bg0":    "#f8f9ff",
#         "bg1":    "#eef0ff",
#         "bg2":    "#e4e7ff",
#         "bg3":    "#d8dcff",
#         "accent": "#6d28d9",
#         "accent2":"#7c3aed",
#         "accent3":"#8b5cf6",
#         "accent4":"#1e1b4b",
#         "accent5":"#ede9fe",
#         "green":  "#16a34a",
#         "red":    "#dc2626",
#         "yellow": "#d97706",
#         "cyan":   "#0891b2",
#         "text":   "#1e1b4b",
#         "text2":  "#4c1d95",
#         "text3":  "#7c3aed",
#         "border": "rgba(109,40,217,0.25)",
#     },
# }

# CURRENT_THEME = "Dark Purple"

# def T(key):
#     """Get a color from the current theme."""
#     return THEMES[CURRENT_THEME][key]


# def make_stylesheet(theme_name: str) -> str:
#     t = THEMES[theme_name]
#     return f"""
#     QMainWindow, QWidget#centralWidget {{
#         background-color: {t['bg0']};
#         color: {t['text']};
#         font-family: 'Space Mono', 'Courier New', monospace;
#     }}
#     QWidget {{
#         background-color: transparent;
#         color: {t['text']};
#         font-family: 'Space Mono', 'Courier New', monospace;
#     }}
#     QFrame#panel {{
#         background-color: {t['bg1']};
#         border: 1px solid {t['accent']}40;
#         border-radius: 14px;
#     }}
#     QFrame#panelHead {{
#         background-color: {t['bg2']};
#         border-bottom: 1px solid {t['accent']}40;
#         border-radius: 0px;
#     }}
#     QLabel#panelTitle {{
#         color: {t['accent3']};
#         font-size: 11px;
#         font-weight: bold;
#         letter-spacing: 2px;
#         text-transform: uppercase;
#     }}
#     QPushButton {{
#         background-color: {t['bg2']};
#         color: {t['text']};
#         border: 1px solid {t['accent']}60;
#         border-radius: 8px;
#         padding: 8px 16px;
#         font-family: 'Space Mono', 'Courier New', monospace;
#         font-size: 12px;
#     }}
#     QPushButton:hover {{
#         background-color: {t['bg3']};
#         border-color: {t['accent']};
#         color: {t['accent3']};
#     }}
#     QPushButton:pressed {{
#         background-color: {t['accent5']};
#     }}
#     QPushButton#btnPrimary {{
#         background-color: {t['accent5']};
#         border-color: {t['accent']};
#         color: {t['accent4']};
#     }}
#     QPushButton#btnPrimary:hover {{
#         background-color: {t['accent']};
#     }}
#     QPushButton#btnDanger {{
#         background-color: {t['red']}20;
#         border-color: {t['red']}60;
#         color: {t['red']};
#     }}
#     QPushButton#btnDanger:hover {{
#         background-color: {t['red']}40;
#     }}
#     QTabWidget::pane {{
#         border: none;
#         background: transparent;
#     }}
#     QTabBar::tab {{
#         background: transparent;
#         color: {t['text3']};
#         border-bottom: 2px solid transparent;
#         padding: 8px 14px;
#         font-size: 11px;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#     }}
#     QTabBar::tab:selected {{
#         color: {t['accent3']};
#         border-bottom-color: {t['accent2']};
#     }}
#     QTabBar::tab:hover {{
#         color: {t['text2']};
#     }}
#     QTextEdit {{
#         background-color: {t['bg2']};
#         color: {t['text2']};
#         border: none;
#         font-family: 'Space Mono', 'Courier New', monospace;
#         font-size: 11px;
#     }}
#     QScrollBar:vertical {{
#         background: {t['bg2']};
#         width: 4px;
#         border-radius: 2px;
#     }}
#     QScrollBar::handle:vertical {{
#         background: {t['accent5']};
#         border-radius: 2px;
#     }}
#     QComboBox {{
#         background-color: {t['bg2']};
#         color: {t['text']};
#         border: 1px solid {t['accent']}60;
#         border-radius: 6px;
#         padding: 4px 10px;
#         font-size: 12px;
#     }}
#     QComboBox::drop-down {{
#         border: none;
#     }}
#     QComboBox QAbstractItemView {{
#         background-color: {t['bg2']};
#         color: {t['text']};
#         selection-background-color: {t['accent5']};
#         border: 1px solid {t['accent']}60;
#     }}
#     QProgressBar {{
#         background-color: {t['bg3']};
#         border: none;
#         border-radius: 3px;
#         height: 6px;
#         text-align: center;
#     }}
#     QProgressBar::chunk {{
#         border-radius: 3px;
#         background-color: {t['accent']};
#     }}
#     """


# # ══════════════════════════════════════════════════════════════════════════════
# #  INFERENCE WORKER THREAD
# # ══════════════════════════════════════════════════════════════════════════════
# class InferenceWorker(QThread):
#     """
#     Runs AttentionEngine in a background thread.
#     Emits state_update(dict) every frame.
#     """
#     state_update = pyqtSignal(dict)
#     frame_update = pyqtSignal(np.ndarray)
#     log_message  = pyqtSignal(str, str)   # (message, type)

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._running = False
#         self._paused  = False
#         self._cap     = None
#         self._engine  = None
#         self._demo_t  = 0.0

#     def run(self):
#         self._running = True

#         if ENGINE_AVAILABLE:
#             self._engine = AttentionEngine()
#             self._engine.load(log_fn=lambda m: self.log_message.emit(m, "info"))
#         else:
#             self.log_message.emit("Demo mode — main.py not found", "warn")

#         self._cap = cv2.VideoCapture(0)
#         if not self._cap.isOpened():
#             self.log_message.emit("Camera not available — demo mode", "warn")
#             self._cap = None

#         while self._running:
#             if self._paused:
#                 self.msleep(50)
#                 continue

#             if self._cap and self._cap.isOpened():
#                 ret, frame = self._cap.read()
#                 if not ret:
#                     self.msleep(30)
#                     continue

#                 if ENGINE_AVAILABLE and self._engine:
#                     state = self._engine.process(frame)
#                 else:
#                     state = self._demo_state()

#                 self.frame_update.emit(frame.copy())
#                 self.state_update.emit(state)
#             else:
#                 # Pure demo mode: no camera
#                 state = self._demo_state()
#                 frame = self._demo_frame(state)
#                 self.frame_update.emit(frame)
#                 self.state_update.emit(state)
#                 self.msleep(33)

#         if self._cap:
#             self._cap.release()

#     def pause(self):
#         self._paused = True

#     def resume(self):
#         self._paused = False

#     def stop(self):
#         self._running = False
#         self.wait()

#     def _demo_state(self) -> dict:
#         self._demo_t += 0.033
#         t = self._demo_t
#         gc = max(0.0, min(1.0, 0.68 + 0.28 * math.sin(t * 0.32)))
#         pc = max(0.0, min(1.0, 0.74 + 0.22 * math.cos(t * 0.26)))
#         gl = "Distracted" if gc > 0.82 else "Focused"
#         pl = "Distracted" if pc > 0.82 else "Focused"
#         focused = (gl == "Focused") and (pl == "Focused")
#         return {
#             "gaze_label": gl, "gaze_conf": gc,
#             "pose_label": pl, "pose_conf": pc,
#             "df_label":   "Focused" if focused else "Distracted",
#             "df_emotion": "neutral",
#             "focused":    focused,
#             "dist":       62 + 15 * math.sin(t * 0.1),
#             "frame":      int(t * 30),
#             "yaw":        0.04 * math.sin(t * 0.2),
#             "pitch":      0.02 * math.cos(t * 0.15),
#             "roll":       0.01 * math.sin(t * 0.1),
#             "gaze_nx":    max(0.0, min(1.0, 0.5 + 0.15 * math.sin(t * 0.3))),
#             "gaze_ny":    max(0.0, min(1.0, 0.5 + 0.1  * math.cos(t * 0.25))),
#         }

#     def _demo_frame(self, state: dict) -> np.ndarray:
#         frame = np.zeros((480, 640, 3), dtype=np.uint8)
#         frame[:] = (12, 8, 20)
#         focused = state["focused"]
#         color = (34, 197, 94) if focused else (239, 68, 68)
#         cv2.ellipse(frame, (320, 200), (80, 100), 0, 0, 360, color, 2)
#         # iris dots
#         eye_off = int(15 * state.get("gaze_nx", 0.5) - 7)
#         cv2.circle(frame, (300 + eye_off, 185), 8, (192, 132, 252), -1)
#         cv2.circle(frame, (340 + eye_off, 185), 8, (192, 132, 252), -1)
#         cv2.putText(frame, f"DEMO | {'FOCUSED' if focused else 'DISTRACTED'}",
#                     (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         gx = int(state.get("gaze_nx", 0.5) * 640)
#         gy = int(state.get("gaze_ny", 0.5) * 480)
#         cv2.circle(frame, (gx, gy), 12, (34, 197, 94), 2)
#         cv2.circle(frame, (gx, gy), 3, (34, 197, 94), -1)
#         return frame


# # ══════════════════════════════════════════════════════════════════════════════
# #  DISTRACTION ENFORCER  (background detection + close other apps + countdown)
# # ══════════════════════════════════════════════════════════════════════════════
# class DistractionEnforcer(QThread):
#     """
#     Monitors if the NeuralEye window is the active/foreground window.
#     If backgrounded for more than 1 second:
#       1. Close all other visible user applications.
#       2. Bring NeuralEye back to front.
#       3. Start a 15-second countdown overlay.
#     """
#     show_countdown  = pyqtSignal(int)   # seconds remaining
#     hide_countdown  = pyqtSignal()
#     bring_to_front  = pyqtSignal()
#     log_event       = pyqtSignal(str, str)

#     def __init__(self, app_title: str, parent=None):
#         super().__init__(parent)
#         self._running    = True
#         self._app_title  = app_title
#         self._countdown  = 0
#         self._in_warning = False

#     def run(self):
#         bg_since = None
#         while self._running:
#             is_fg = self._is_foreground()
#             if not is_fg:
#                 if bg_since is None:
#                     bg_since = time.time()
#                 elapsed = time.time() - bg_since
#                 if elapsed >= 1.0 and not self._in_warning:
#                     self._in_warning = True
#                     self.log_event.emit("App went to background — enforcing focus!", "warn")
#                     self._close_other_apps()
#                     self.bring_to_front.emit()
#                     self._run_countdown()
#             else:
#                 if bg_since is not None:
#                     bg_since = None
#                 if self._in_warning:
#                     self._in_warning = False
#                     self.hide_countdown.emit()
#             self.msleep(500)

#     def _run_countdown(self):
#         for sec in range(WARNING_INTERVAL_S, 0, -1):
#             if not self._running:
#                 break
#             self.show_countdown.emit(sec)
#             self.msleep(1000)
#         self.hide_countdown.emit()
#         self._in_warning = False
#         self.bring_to_front.emit()

#     def _is_foreground(self) -> bool:
#         try:
#             system = platform.system()
#             if system == "Windows":
#                 import ctypes
#                 hwnd = ctypes.windll.user32.GetForegroundWindow()
#                 length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
#                 buf = ctypes.create_unicode_buffer(length + 1)
#                 ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
#                 return self._app_title.lower() in buf.value.lower()
#             elif system == "Darwin":
#                 result = subprocess.run(
#                     ["osascript", "-e",
#                      'tell application "System Events" to get name of first process whose frontmost is true'],
#                     capture_output=True, text=True, timeout=2
#                 )
#                 return "python" in result.stdout.lower() or "neuraleye" in result.stdout.lower()
#             else:
#                 result = subprocess.run(
#                     ["xdotool", "getactivewindow", "getwindowname"],
#                     capture_output=True, text=True, timeout=2
#                 )
#                 return self._app_title.lower() in result.stdout.lower() or \
#                        "neuraleye" in result.stdout.lower() or \
#                        "python" in result.stdout.lower()
#         except Exception:
#             return True  # assume foreground if we can't check

#     def _close_other_apps(self):
#         """
#         Close visible, non-system user windows except ourselves.
#         """
#         system = platform.system()
#         try:
#             if system == "Windows":
#                 self._close_other_apps_windows()
#             elif system == "Darwin":
#                 self._close_other_apps_mac()
#             else:
#                 self._close_other_apps_linux()
#         except Exception as exc:
#             self.log_event.emit(f"Could not close apps: {exc}", "warn")

#     def _close_other_apps_windows(self):
#         import ctypes
#         import psutil
#         SKIP = {"neuraleye", "python", "pythonw", "explorer", "taskmgr",
#                 "winlogon", "csrss", "svchost", "dwm", "system idle",
#                 "registry", "lsass", "services", "smss", "wininit"}
#         for proc in psutil.process_iter(['pid', 'name', 'status']):
#             try:
#                 name = proc.info['name'].lower().replace('.exe', '')
#                 if proc.info['status'] != 'running':
#                     continue
#                 if any(s in name for s in SKIP):
#                     continue
#                 # Only terminate processes that have windows
#                 proc.terminate()
#                 self.log_event.emit(f"Closed: {proc.info['name']}", "warn")
#             except Exception:
#                 continue

#     def _close_other_apps_mac(self):
#         SKIP = {"finder", "dock", "loginwindow", "systemuiserver",
#                 "spotlight", "neuraleye", "python", "python3"}
#         result = subprocess.run(
#             ["osascript", "-e",
#              'tell application "System Events" to get name of every process whose background only is false'],
#             capture_output=True, text=True, timeout=5
#         )
#         apps = [a.strip() for a in result.stdout.split(",")]
#         for app in apps:
#             if not app:
#                 continue
#             app_low = app.lower()
#             if any(s in app_low for s in SKIP):
#                 continue
#             try:
#                 subprocess.run(
#                     ["osascript", "-e", f'tell application "{app}" to quit'],
#                     timeout=3, capture_output=True
#                 )
#                 self.log_event.emit(f"Closed: {app}", "warn")
#             except Exception:
#                 continue

#     def _close_other_apps_linux(self):
#         import psutil
#         SKIP = {"neuraleye", "python", "python3", "bash", "sh", "zsh",
#                 "gdm", "gdm3", "lightdm", "Xorg", "xfwm4", "gnome-shell",
#                 "kwin", "mutter", "dbus", "systemd", "pulseaudio"}
#         for proc in psutil.process_iter(['pid', 'name', 'status']):
#             try:
#                 name = proc.info['name'].lower()
#                 if proc.info['status'] != 'running':
#                     continue
#                 if any(s.lower() in name for s in SKIP):
#                     continue
#                 if name in ('python', 'python3'):
#                     continue
#                 proc.terminate()
#                 self.log_event.emit(f"Closed: {proc.info['name']}", "warn")
#             except Exception:
#                 continue

#     def stop(self):
#         self._running = False
#         self.wait()


# # ══════════════════════════════════════════════════════════════════════════════
# #  GAZE SCREEN OVERLAY  (translucent dot showing where user is looking)
# # ══════════════════════════════════════════════════════════════════════════════
# class GazeOverlay(QWidget):
#     """
#     A transparent topmost window that draws a glowing dot
#     at the estimated gaze position on the screen.
#     """
#     def __init__(self):
#         super().__init__()
#         self.setWindowFlags(
#             Qt.WindowType.FramelessWindowHint |
#             Qt.WindowType.WindowStaysOnTopHint |
#             Qt.WindowType.Tool |
#             Qt.WindowType.WindowTransparentForInput
#         )
#         self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
#         self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
#         self._gx = 0.5
#         self._gy = 0.5
#         screen = QApplication.primaryScreen().geometry()
#         self.setGeometry(screen)
#         self.show()

#     def update_gaze(self, nx: float, ny: float):
#         self._gx = nx
#         self._gy = ny
#         self.update()

#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#         screen = QApplication.primaryScreen().geometry()
#         gx = int(self._gx * screen.width())
#         gy = int(self._gy * screen.height())

#         # Outer glow
#         for r in [28, 20, 14]:
#             alpha = int(80 * (1 - r / 30))
#             painter.setPen(Qt.PenStyle.NoPen)
#             painter.setBrush(QBrush(QColor(34, 197, 94, alpha)))
#             painter.drawEllipse(QPoint(gx, gy), r, r)

#         # Inner dot
#         painter.setBrush(QBrush(QColor(34, 197, 94, 200)))
#         painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
#         painter.drawEllipse(QPoint(gx, gy), 7, 7)

#         painter.end()


# # ══════════════════════════════════════════════════════════════════════════════
# #  COUNTDOWN OVERLAY
# # ══════════════════════════════════════════════════════════════════════════════
# class CountdownOverlay(QWidget):
#     """Full-screen translucent countdown overlay."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowFlags(
#             Qt.WindowType.FramelessWindowHint |
#             Qt.WindowType.WindowStaysOnTopHint |
#             Qt.WindowType.Tool
#         )
#         self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
#         self._seconds = 15
#         screen = QApplication.primaryScreen().geometry()
#         self.setGeometry(screen)
#         self.hide()

#     def show_seconds(self, sec: int):
#         self._seconds = sec
#         self.update()
#         if not self.isVisible():
#             self.show()
#             self.raise_()

#     def hide_overlay(self):
#         self.hide()

#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#         w, h = self.width(), self.height()

#         # Dark overlay
#         painter.fillRect(0, 0, w, h, QColor(7, 6, 13, 210))

#         # Warning box
#         box_w, box_h = 420, 280
#         bx = (w - box_w) // 2
#         by = (h - box_h) // 2
#         painter.setBrush(QBrush(QColor(20, 17, 42)))
#         painter.setPen(QPen(QColor(239, 68, 68, 180), 2))
#         painter.drawRoundedRect(bx, by, box_w, box_h, 16, 16)

#         # Title
#         painter.setPen(QColor(239, 68, 68))
#         painter.setFont(QFont("Space Mono", 14, QFont.Weight.Bold))
#         painter.drawText(QRect(bx, by + 30, box_w, 40),
#                          Qt.AlignmentFlag.AlignCenter, "⚠  DISTRACTION DETECTED")

#         # Subtitle
#         painter.setPen(QColor(167, 139, 202))
#         painter.setFont(QFont("Space Mono", 10))
#         painter.drawText(QRect(bx, by + 78, box_w, 30),
#                          Qt.AlignmentFlag.AlignCenter, "Returning to NeuralEye in...")

#         # Countdown number
#         painter.setPen(QColor(239, 68, 68))
#         painter.setFont(QFont("Space Mono", 64, QFont.Weight.Bold))
#         painter.drawText(QRect(bx, by + 100, box_w, 110),
#                          Qt.AlignmentFlag.AlignCenter, str(self._seconds))

#         # Sub-text
#         painter.setPen(QColor(107, 77, 140))
#         painter.setFont(QFont("Space Mono", 9))
#         painter.drawText(QRect(bx, by + 220, box_w, 40),
#                          Qt.AlignmentFlag.AlignCenter,
#                          "Other applications have been closed for focus.")

#         painter.end()


# # ══════════════════════════════════════════════════════════════════════════════
# #  CUSTOM WIDGETS
# # ══════════════════════════════════════════════════════════════════════════════

# class MetricCard(QFrame):
#     def __init__(self, label: str, value: str, sub: str = "", parent=None):
#         super().__init__(parent)
#         self.setObjectName("panel")
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(14, 12, 14, 12)
#         layout.setSpacing(4)

#         self._lbl = QLabel(label.upper())
#         self._lbl.setStyleSheet("font-size:10px; letter-spacing:2px; color: palette(dark);")

#         self._val = QLabel(value)
#         self._val.setStyleSheet("font-size:26px; font-weight:bold;")

#         self._sub = QLabel(sub)
#         self._sub.setStyleSheet("font-size:11px;")

#         layout.addWidget(self._lbl)
#         layout.addWidget(self._val)
#         layout.addWidget(self._sub)

#     def set_value(self, v: str):
#         self._val.setText(v)

#     def set_sub(self, s: str):
#         self._sub.setText(s)

#     def set_color(self, color: str):
#         self._val.setStyleSheet(f"font-size:26px; font-weight:bold; color:{color};")

#     def set_sub_color(self, color: str):
#         self._sub.setStyleSheet(f"font-size:11px; color:{color};")

#     def apply_theme(self, t: dict):
#         self._lbl.setStyleSheet(f"font-size:10px; letter-spacing:2px; color:{t['text3']};")
#         self._sub.setStyleSheet(f"font-size:11px; color:{t['text3']};")
#         self.setStyleSheet(
#             f"QFrame#panel{{background:{t['bg2']};border:1px solid {t['accent']}40;border-radius:10px;}}"
#         )


# class SignalBar(QWidget):
#     def __init__(self, label: str, parent=None):
#         super().__init__(parent)
#         layout = QHBoxLayout(self)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(8)

#         self._lbl = QLabel(label.upper())
#         self._lbl.setFixedWidth(52)
#         self._lbl.setStyleSheet("font-size:10px; letter-spacing:1px;")

#         self._bar = QProgressBar()
#         self._bar.setRange(0, 100)
#         self._bar.setValue(80)
#         self._bar.setTextVisible(False)
#         self._bar.setFixedHeight(6)
#         self._bar.setStyleSheet("QProgressBar::chunk{background:#22c55e;}")

#         self._val = QLabel("FOCUSED 0.88")
#         self._val.setFixedWidth(110)
#         self._val.setAlignment(Qt.AlignmentFlag.AlignRight)
#         self._val.setStyleSheet("font-size:10px;")

#         layout.addWidget(self._lbl)
#         layout.addWidget(self._bar, 1)
#         layout.addWidget(self._val)

#     def set_value(self, pct: int, label_str: str, focused: bool):
#         self._bar.setValue(pct)
#         color = "#22c55e" if focused else "#ef4444"
#         self._bar.setStyleSheet(
#             f"QProgressBar{{background:#1c1836;border:none;border-radius:3px;height:6px;}}"
#             f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
#         )
#         self._val.setText(label_str)


# class CameraWidget(QLabel):
#     """Displays the camera/demo frame with gaze point overlay."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumSize(320, 240)
#         self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.setStyleSheet("background:#07060d; border-radius:10px;")
#         self._gx = 0.5
#         self._gy = 0.5
#         self._frame = None

#     def update_frame(self, frame: np.ndarray, gaze_nx: float = 0.5, gaze_ny: float = 0.5):
#         self._gx = gaze_nx
#         self._gy = gaze_ny
#         self._frame = frame
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
#         pix = QPixmap.fromImage(qt_img).scaled(
#             self.width(), self.height(),
#             Qt.AspectRatioMode.KeepAspectRatio,
#             Qt.TransformationMode.SmoothTransformation
#         )
#         self.setPixmap(pix)


# class DistanceRing(QWidget):
#     """SVG-style distance ring drawn with QPainter."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumSize(100, 100)
#         self._dist = 62.0
#         self._theme = THEMES[CURRENT_THEME]

#     def set_distance(self, d: float):
#         self._dist = d
#         self.update()

#     def apply_theme(self, t: dict):
#         self._theme = t
#         self.update()

#     def paintEvent(self, event):
#         t = self._theme
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#         w, h = self.width(), self.height()
#         r = min(w, h) // 2 - 10
#         cx, cy = w // 2, h // 2

#         # Background circle
#         painter.setPen(QPen(QColor(t["bg3"]), 8))
#         painter.setBrush(Qt.BrushStyle.NoBrush)
#         painter.drawEllipse(QPoint(cx, cy), r, r)

#         # Arc
#         pct = max(0.0, min(1.0, self._dist / 90.0))
#         span = int(pct * 360 * 16)
#         if self._dist > 90:
#             arc_color = QColor(t["red"])
#         elif self._dist < 30:
#             arc_color = QColor(t["yellow"])
#         else:
#             arc_color = QColor(t["accent"])

#         pen = QPen(arc_color, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
#         painter.setPen(pen)
#         painter.drawArc(cx - r, cy - r, r * 2, r * 2, 90 * 16, -span)

#         # Center text
#         painter.setPen(QColor(t["accent4"]))
#         painter.setFont(QFont("Space Mono", 14, QFont.Weight.Bold))
#         painter.drawText(QRect(cx - r, cy - 14, r * 2, 30),
#                          Qt.AlignmentFlag.AlignCenter, str(int(self._dist)))
#         painter.setFont(QFont("Space Mono", 8))
#         painter.setPen(QColor(t["text3"]))
#         painter.drawText(QRect(cx - r, cy + 10, r * 2, 20),
#                          Qt.AlignmentFlag.AlignCenter, "cm")
#         painter.end()


# class HeadPoseRadar(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumSize(140, 140)
#         self._yaw = 0.0
#         self._pitch = 0.0
#         self._roll = 0.0
#         self._theme = THEMES[CURRENT_THEME]

#     def set_pose(self, yaw: float, pitch: float, roll: float):
#         self._yaw = yaw
#         self._pitch = pitch
#         self._roll = roll
#         self.update()

#     def apply_theme(self, t: dict):
#         self._theme = t
#         self.update()

#     def paintEvent(self, event):
#         t = self._theme
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#         w, h = self.width(), self.height()
#         cx, cy = w // 2, h // 2
#         r = min(w, h) // 2 - 20

#         # Grid rings
#         for sc in [0.33, 0.66, 1.0]:
#             painter.setPen(QPen(QColor(t["accent"] + "40"), 1))
#             painter.setBrush(Qt.BrushStyle.NoBrush)
#             painter.drawEllipse(QPoint(cx, cy), int(r * sc), int(r * sc))

#         # Axes
#         axes = [(0, -r), (int(r * 0.87), int(r * 0.5)), (-int(r * 0.87), int(r * 0.5))]
#         labels = ["YAW", "PITCH", "ROLL"]
#         values = [self._yaw, self._pitch, self._roll]
#         clamp_ranges = [(0.4, 0.8), (0.3, 0.6), (0.2, 0.4)]

#         pts = []
#         for (ax, ay), val, (lo, hi) in zip(axes, values, clamp_ranges):
#             painter.setPen(QPen(QColor(t["accent"] + "50"), 1))
#             painter.drawLine(cx, cy, cx + ax, cy + ay)
#             n = max(0.0, min(1.0, (abs(val) + lo) / (lo + hi)))
#             pts.append((cx + int(ax * n), cy + int(ay * n)))

#         # Filled polygon
#         from PyQt6.QtGui import QPolygon
#         poly = QPolygon([QPoint(x, y) for x, y in pts])
#         painter.setPen(QPen(QColor(t["accent2"]), 1.5))
#         painter.setBrush(QBrush(QColor(t["accent2"] + "40")))
#         painter.drawPolygon(poly)

#         # Points
#         for px, py in pts:
#             painter.setBrush(QBrush(QColor(t["accent3"])))
#             painter.setPen(Qt.PenStyle.NoPen)
#             painter.drawEllipse(QPoint(px, py), 4, 4)

#         # Labels
#         painter.setPen(QColor(t["text3"]))
#         painter.setFont(QFont("Space Mono", 7))
#         lbl_offsets = [(0, -r - 14), (int(r * 0.87) + 8, int(r * 0.5) + 14),
#                        (-int(r * 0.87) - 8, int(r * 0.5) + 14)]
#         for lbl, (ox, oy) in zip(labels, lbl_offsets):
#             painter.drawText(QRect(cx + ox - 20, cy + oy - 8, 40, 16),
#                              Qt.AlignmentFlag.AlignCenter, lbl)
#         painter.end()


# class AttentionChart(QWidget):
#     """Scrolling line chart for gaze + pose confidence."""
#     HISTORY = 60

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumHeight(160)
#         self._gaze = [0.8] * self.HISTORY
#         self._pose = [0.8] * self.HISTORY
#         self._theme = THEMES[CURRENT_THEME]

#     def push(self, gaze: float, pose: float):
#         self._gaze.append(gaze)
#         self._gaze = self._gaze[-self.HISTORY:]
#         self._pose.append(pose)
#         self._pose = self._pose[-self.HISTORY:]
#         self.update()

#     def apply_theme(self, t: dict):
#         self._theme = t
#         self.update()

#     def paintEvent(self, event):
#         t = self._theme
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#         w, h = self.width(), self.height()

#         # Background
#         painter.fillRect(0, 0, w, h, QColor(t["bg1"]))

#         # Grid lines
#         for i in range(5):
#             y = int(h - (i / 4) * h)
#             painter.setPen(QPen(QColor(t["accent"] + "20"), 1))
#             painter.drawLine(0, y, w, y)

#         # Y-axis labels
#         painter.setFont(QFont("Space Mono", 8))
#         painter.setPen(QColor(t["text3"]))
#         for i in range(5):
#             y = int(h - (i / 4) * h)
#             painter.drawText(2, y - 2, f"{i/4:.1f}")

#         def draw_line(data, color_hex, fill_hex):
#             from PyQt6.QtGui import QPainterPath
#             n = len(data)
#             path = QPainterPath()
#             fill = QPainterPath()
#             for idx, v in enumerate(data):
#                 x = int(idx / (n - 1) * w)
#                 y = int(h - v * h)
#                 if idx == 0:
#                     path.moveTo(x, y)
#                     fill.moveTo(x, h)
#                     fill.lineTo(x, y)
#                 else:
#                     path.lineTo(x, y)
#                     fill.lineTo(x, y)
#             fill.lineTo(w, h)
#             fill.lineTo(0, h)
#             fill.closeSubpath()
#             painter.setPen(Qt.PenStyle.NoPen)
#             painter.setBrush(QBrush(QColor(fill_hex)))
#             painter.drawPath(fill)
#             painter.setPen(QPen(QColor(color_hex), 1.5))
#             painter.setBrush(Qt.BrushStyle.NoBrush)
#             painter.drawPath(path)

#         draw_line(self._gaze, t["green"], t["green"] + "18")
#         draw_line(self._pose, t["accent2"], t["accent2"] + "18")
#         painter.end()


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN WINDOW
# # ══════════════════════════════════════════════════════════════════════════════
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("NeuralEye — Attention Tracking System")
#         self.setMinimumSize(1100, 700)
#         self.resize(1350, 820)

#         self._session_sec   = 0
#         self._distract_cnt  = 0
#         self._focus_rate    = 82
#         self._focus_history = []
#         self._dist_min = 99999.0
#         self._dist_max = 0.0
#         self._cam_active    = False
#         self._session_running = True
#         self._current_theme   = "Dark Purple"

#         self._build_ui()
#         self._apply_theme("Dark Purple")

#         # Timers
#         self._clock_timer = QTimer(self)
#         self._clock_timer.timeout.connect(self._tick_clock)
#         self._clock_timer.start(1000)

#         self._session_timer = QTimer(self)
#         self._session_timer.timeout.connect(self._tick_session)
#         self._session_timer.start(1000)

#         # Inference worker
#         self._worker = InferenceWorker()
#         self._worker.state_update.connect(self._on_state)
#         self._worker.frame_update.connect(self._on_frame)
#         self._worker.log_message.connect(self._append_log)
#         self._worker.start()
#         self._cam_active = True
#         self._btn_cam.setText("⏸  Pause Camera")

#         # Gaze overlay
#         self._gaze_overlay = GazeOverlay()

#         # Countdown overlay
#         self._countdown_overlay = CountdownOverlay()

#         # Distraction enforcer
#         self._enforcer = DistractionEnforcer("NeuralEye")
#         self._enforcer.show_countdown.connect(self._countdown_overlay.show_seconds)
#         self._enforcer.hide_countdown.connect(self._countdown_overlay.hide_overlay)
#         self._enforcer.bring_to_front.connect(self._force_foreground)
#         self._enforcer.log_event.connect(self._append_log)
#         self._enforcer.start()

#     # ── UI CONSTRUCTION ───────────────────────────────────────────────────────
#     def _build_ui(self):
#         central = QWidget()
#         central.setObjectName("centralWidget")
#         self.setCentralWidget(central)
#         root = QVBoxLayout(central)
#         root.setContentsMargins(20, 16, 20, 16)
#         root.setSpacing(14)

#         root.addWidget(self._make_header())
#         root.addWidget(self._make_metrics_bar())

#         body = QHBoxLayout()
#         body.setSpacing(16)
#         body.addLayout(self._make_left_col(), stretch=3)
#         body.addLayout(self._make_right_col(), stretch=5)
#         root.addLayout(body, stretch=1)

#         root.addWidget(self._make_footer())

#     def _make_header(self) -> QWidget:
#         w = QWidget()
#         layout = QHBoxLayout(w)
#         layout.setContentsMargins(0, 0, 0, 12)

#         # Logo
#         logo_icon = QLabel("👁")
#         logo_icon.setStyleSheet(
#             "font-size:24px; background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
#             "stop:0 #4c1d95,stop:1 #7c3aed); border-radius:10px; "
#             "padding:6px 10px;"
#         )

#         logo_text = QVBoxLayout()
#         title = QLabel("NeuralEye")
#         title.setStyleSheet("font-size:20px; font-weight:800; letter-spacing:-0.5px;")
#         sub = QLabel("ATTENTION TRACKING SYSTEM v2.0")
#         sub.setStyleSheet("font-size:10px; letter-spacing:2px;")
#         logo_text.addWidget(title)
#         logo_text.addWidget(sub)
#         logo_text.setSpacing(2)

#         logo_w = QWidget()
#         logo_l = QHBoxLayout(logo_w)
#         logo_l.setContentsMargins(0, 0, 0, 0)
#         logo_l.setSpacing(12)
#         logo_l.addWidget(logo_icon)
#         logo_l.addLayout(logo_text)

#         layout.addWidget(logo_w)
#         layout.addStretch(1)

#         # Theme selector
#         theme_lbl = QLabel("THEME:")
#         theme_lbl.setStyleSheet("font-size:10px; letter-spacing:1px;")
#         self._theme_combo = QComboBox()
#         for name in THEMES:
#             self._theme_combo.addItem(name)
#         self._theme_combo.setCurrentText("Dark Purple")
#         self._theme_combo.currentTextChanged.connect(self._apply_theme)
#         self._theme_combo.setFixedWidth(160)

#         layout.addWidget(theme_lbl)
#         layout.addWidget(self._theme_combo)
#         layout.setSpacing(16)

#         self._clock_lbl = QLabel("--:--:--")
#         self._clock_lbl.setStyleSheet("font-size:13px; letter-spacing:1px;")

#         badge = QFrame()
#         badge.setObjectName("statusBadge")
#         badge_l = QHBoxLayout(badge)
#         badge_l.setContentsMargins(10, 6, 10, 6)
#         badge_l.setSpacing(8)
#         self._sys_dot = QLabel("●")
#         self._sys_dot.setStyleSheet("font-size:10px; color:#22c55e;")
#         self._sys_status = QLabel("SYSTEM ONLINE")
#         self._sys_status.setStyleSheet("font-size:12px;")
#         badge_l.addWidget(self._sys_dot)
#         badge_l.addWidget(self._sys_status)
#         badge.setStyleSheet(
#             "QFrame#statusBadge{border:1px solid rgba(124,58,237,0.4);"
#             "border-radius:8px; padding:2px;}"
#         )

#         layout.addWidget(self._clock_lbl)
#         layout.addWidget(badge)
#         return w

#     def _make_metrics_bar(self) -> QWidget:
#         w = QWidget()
#         layout = QHBoxLayout(w)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(12)

#         self._card_focus   = MetricCard("Focus Rate",         "82%",   "↑ +4% vs last min")
#         self._card_distract= MetricCard("Distraction Events", "0",     "this session")
#         self._card_dist    = MetricCard("Avg Distance",        "62 cm", "optimal: 50–80 cm")
#         self._card_session = MetricCard("Session Time",        "00:00", "active session")

#         for card in [self._card_focus, self._card_distract,
#                      self._card_dist, self._card_session]:
#             layout.addWidget(card, stretch=1)

#         return w

#     def _make_left_col(self) -> QVBoxLayout:
#         col = QVBoxLayout()
#         col.setSpacing(12)

#         # Camera panel
#         cam_panel = QFrame()
#         cam_panel.setObjectName("panel")
#         cam_l = QVBoxLayout(cam_panel)
#         cam_l.setContentsMargins(0, 0, 0, 0)
#         cam_l.setSpacing(0)

#         cam_head = QFrame()
#         cam_head.setObjectName("panelHead")
#         cam_head_l = QHBoxLayout(cam_head)
#         cam_head_l.setContentsMargins(14, 10, 14, 10)
#         cam_title = QLabel("CAMERA FEED")
#         cam_title.setObjectName("panelTitle")
#         self._fps_lbl = QLabel("30 FPS")
#         self._fps_lbl.setStyleSheet("font-size:10px; letter-spacing:1px;")
#         cam_head_l.addWidget(cam_title)
#         cam_head_l.addStretch()
#         cam_head_l.addWidget(self._fps_lbl)

#         self._cam_widget = CameraWidget()
#         self._cam_widget.setMinimumHeight(240)

#         cam_l.addWidget(cam_head)
#         cam_l.addWidget(self._cam_widget, stretch=1)
#         col.addWidget(cam_panel)

#         # Status panel
#         status_panel = QFrame()
#         status_panel.setObjectName("panel")
#         status_l = QVBoxLayout(status_panel)
#         status_l.setContentsMargins(0, 0, 0, 0)
#         status_l.setSpacing(0)

#         # Status big
#         self._status_big_lbl = QLabel("CURRENT STATUS")
#         self._status_big_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self._status_big_lbl.setStyleSheet("font-size:9px; letter-spacing:3px; padding-top:12px;")

#         self._status_value = QLabel("FOCUSED")
#         self._status_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self._status_value.setStyleSheet(
#             "font-size:30px; font-weight:800; color:#22c55e; padding:6px;"
#         )

#         conf_w = QWidget()
#         conf_l = QVBoxLayout(conf_w)
#         conf_l.setContentsMargins(14, 4, 14, 12)
#         conf_l.setSpacing(4)
#         conf_row = QHBoxLayout()
#         lbl_conf = QLabel("Confidence")
#         lbl_conf.setStyleSheet("font-size:10px;")
#         self._conf_pct = QLabel("87%")
#         self._conf_pct.setStyleSheet("font-size:10px;")
#         conf_row.addWidget(lbl_conf)
#         conf_row.addStretch()
#         conf_row.addWidget(self._conf_pct)
#         self._conf_bar = QProgressBar()
#         self._conf_bar.setRange(0, 100)
#         self._conf_bar.setValue(87)
#         self._conf_bar.setTextVisible(False)
#         self._conf_bar.setFixedHeight(5)
#         conf_l.addLayout(conf_row)
#         conf_l.addWidget(self._conf_bar)

#         # 3 mini-metrics
#         mini_w = QWidget()
#         mini_l = QHBoxLayout(mini_w)
#         mini_l.setContentsMargins(12, 8, 12, 8)
#         mini_l.setSpacing(8)
#         self._m_dist  = self._mini_metric("DIST",     "62cm")
#         self._m_frame = self._mini_metric("FRAME",    "0000")
#         self._m_df    = self._mini_metric("DEEPFACE", "neutral")
#         mini_l.addWidget(self._m_dist)
#         mini_l.addWidget(self._m_frame)
#         mini_l.addWidget(self._m_df)

#         # Signal bars
#         sigs_w = QWidget()
#         sigs_l = QVBoxLayout(sigs_w)
#         sigs_l.setContentsMargins(14, 10, 14, 10)
#         sigs_l.setSpacing(10)
#         self._sig_gaze = SignalBar("Gaze")
#         self._sig_pose = SignalBar("Pose")
#         self._sig_emo  = SignalBar("Emotion")
#         sigs_l.addWidget(self._sig_gaze)
#         sigs_l.addWidget(self._sig_pose)
#         sigs_l.addWidget(self._sig_emo)

#         # Controls
#         ctrl_w = QWidget()
#         ctrl_l = QHBoxLayout(ctrl_w)
#         ctrl_l.setContentsMargins(12, 8, 12, 12)
#         ctrl_l.setSpacing(8)
#         self._btn_cam    = QPushButton("▶  Start Camera")
#         self._btn_cam.setObjectName("btnPrimary")
#         self._btn_export = QPushButton("↓  Export Log")
#         self._btn_stop   = QPushButton("■  Stop")
#         self._btn_stop.setObjectName("btnDanger")
#         self._btn_cam.clicked.connect(self._toggle_camera)
#         self._btn_export.clicked.connect(self._export_log)
#         self._btn_stop.clicked.connect(self._stop_session)
#         ctrl_l.addWidget(self._btn_cam)
#         ctrl_l.addWidget(self._btn_export)
#         ctrl_l.addWidget(self._btn_stop)

#         sep = QFrame()
#         sep.setFrameShape(QFrame.Shape.HLine)
#         sep.setStyleSheet("color: rgba(124,58,237,0.25);")

#         status_l.addWidget(self._status_big_lbl)
#         status_l.addWidget(self._status_value)
#         status_l.addWidget(conf_w)
#         status_l.addWidget(sep)
#         status_l.addWidget(mini_w)
#         sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
#         sep2.setStyleSheet("color: rgba(124,58,237,0.15);")
#         status_l.addWidget(sep2)
#         status_l.addWidget(sigs_w)
#         status_l.addWidget(ctrl_w)

#         col.addWidget(status_panel)
#         return col

#     def _make_right_col(self) -> QVBoxLayout:
#         col = QVBoxLayout()
#         col.setSpacing(12)

#         # Timeline chart
#         chart_panel = QFrame()
#         chart_panel.setObjectName("panel")
#         chart_l = QVBoxLayout(chart_panel)
#         chart_l.setContentsMargins(0, 0, 0, 0)
#         chart_l.setSpacing(0)

#         chart_head = QFrame()
#         chart_head.setObjectName("panelHead")
#         chart_head_l = QHBoxLayout(chart_head)
#         chart_head_l.setContentsMargins(14, 10, 14, 10)
#         chart_title = QLabel("ATTENTION TIMELINE")
#         chart_title.setObjectName("panelTitle")
#         legend_w = QWidget()
#         legend_l = QHBoxLayout(legend_w)
#         legend_l.setContentsMargins(0, 0, 0, 0)
#         legend_l.setSpacing(12)
#         for color, lbl in [("#22c55e", "Gaze"), ("#9d5cf6", "Pose")]:
#             dot = QLabel("■")
#             dot.setStyleSheet(f"color:{color}; font-size:10px;")
#             txt = QLabel(lbl)
#             txt.setStyleSheet("font-size:10px;")
#             legend_l.addWidget(dot)
#             legend_l.addWidget(txt)
#         chart_head_l.addWidget(chart_title)
#         chart_head_l.addStretch()
#         chart_head_l.addWidget(legend_w)

#         self._chart = AttentionChart()
#         chart_l.addWidget(chart_head)
#         chart_l.addWidget(self._chart)
#         col.addWidget(chart_panel)

#         # Tabs panel
#         tabs_panel = QFrame()
#         tabs_panel.setObjectName("panel")
#         tabs_l = QVBoxLayout(tabs_panel)
#         tabs_l.setContentsMargins(0, 0, 0, 0)

#         self._tabs = QTabWidget()
#         self._tabs.setDocumentMode(True)

#         # Emotions tab
#         emo_w = QWidget()
#         emo_grid = QGridLayout(emo_w)
#         emo_grid.setContentsMargins(14, 14, 14, 14)
#         emo_grid.setSpacing(8)
#         self._emo_cards = {}
#         emotions = [("neutral", "😐"), ("happy", "😊"), ("surprise", "😮"),
#                     ("sad", "😔"), ("angry", "😠"), ("fear", "😨")]
#         for i, (name, icon) in enumerate(emotions):
#             card = self._make_emo_card(icon, name, 42 if name == "neutral" else 10)
#             self._emo_cards[name] = card
#             emo_grid.addWidget(card, i // 3, i % 3)

#         # Models tab
#         models_w = QWidget()
#         models_l = QHBoxLayout(models_w)
#         models_l.setContentsMargins(14, 14, 14, 14)
#         models_l.setSpacing(12)
#         models_l.addWidget(self._make_model_card(
#             "Gaze Model",
#             [("Architecture", "Dense 128→64→32"), ("Input features", "43"),
#              ("Train accuracy", "94.2%"), ("Val accuracy", "91.7%"),
#              ("L2 regularizer", "0.001"), ("Dropout", "0.3 / 0.2"),
#              ("Dataset", "RAVDESS landmarks")]
#         ))
#         models_l.addWidget(self._make_model_card(
#             "Pose Model",
#             [("Architecture", "Dense 64→32→16"), ("Input features", "4 (y/p/r/mag)"),
#              ("Train accuracy", "89.8%"), ("Val accuracy", "87.4%"),
#              ("L2 regularizer", "0.001"), ("Dropout", "0.3 / 0.2"),
#              ("Dataset", "AFLW2000-3D")]
#         ))

#         # Log tab
#         log_w = QWidget()
#         log_l = QVBoxLayout(log_w)
#         log_l.setContentsMargins(0, 0, 0, 0)
#         self._log_text = QTextEdit()
#         self._log_text.setReadOnly(True)
#         self._log_text.setMaximumHeight(160)
#         log_l.addWidget(self._log_text)

#         self._tabs.addTab(emo_w,     "Emotions")
#         self._tabs.addTab(models_w,  "Models")
#         self._tabs.addTab(log_w,     "Event Log")

#         tabs_l.addWidget(self._tabs)
#         col.addWidget(tabs_panel)

#         # Distance + Head Pose row
#         bottom_row = QHBoxLayout()
#         bottom_row.setSpacing(12)

#         dist_panel = QFrame()
#         dist_panel.setObjectName("panel")
#         dist_l = QVBoxLayout(dist_panel)
#         dist_l.setContentsMargins(0, 0, 0, 0)
#         dist_head = QFrame(); dist_head.setObjectName("panelHead")
#         dist_hl = QHBoxLayout(dist_head)
#         dist_hl.setContentsMargins(12, 8, 12, 8)
#         dist_hl.addWidget(QLabel("DISTANCE", objectName="panelTitle"))
#         dist_l.addWidget(dist_head)
#         self._dist_ring = DistanceRing()
#         self._dist_ring.setFixedHeight(110)
#         dist_l.addWidget(self._dist_ring)
#         self._dist_status = QLabel("OPTIMAL RANGE")
#         self._dist_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self._dist_status.setStyleSheet("font-size:9px; letter-spacing:2px; padding:4px;")
#         dist_l.addWidget(self._dist_status)
#         dist_mini = QHBoxLayout()
#         dist_mini.setContentsMargins(10, 0, 10, 10)
#         self._dist_min_lbl = QLabel("—")
#         self._dist_max_lbl = QLabel("—")
#         for lbl, caption in [(self._dist_min_lbl, "Min"), (self._dist_max_lbl, "Max")]:
#             c = QWidget()
#             cl = QVBoxLayout(c)
#             cl.setContentsMargins(6, 6, 6, 6)
#             cl.setSpacing(2)
#             cap = QLabel(caption.upper())
#             cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
#             cap.setStyleSheet("font-size:8px; letter-spacing:1px;")
#             lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#             lbl.setStyleSheet("font-size:14px; font-weight:bold;")
#             cl.addWidget(cap)
#             cl.addWidget(lbl)
#             dist_mini.addWidget(c)
#         dist_l.addLayout(dist_mini)

#         pose_panel = QFrame()
#         pose_panel.setObjectName("panel")
#         pose_l = QVBoxLayout(pose_panel)
#         pose_l.setContentsMargins(0, 0, 0, 0)
#         pose_head = QFrame(); pose_head.setObjectName("panelHead")
#         pose_hl = QHBoxLayout(pose_head)
#         pose_hl.setContentsMargins(12, 8, 12, 8)
#         pose_hl.addWidget(QLabel("HEAD POSE", objectName="panelTitle"))
#         pose_l.addWidget(pose_head)
#         self._radar = HeadPoseRadar()
#         self._radar.setFixedHeight(130)
#         pose_l.addWidget(self._radar)
#         pose_vals = QHBoxLayout()
#         pose_vals.setContentsMargins(8, 0, 8, 10)
#         self._yaw_lbl   = QLabel("0.00")
#         self._pitch_lbl = QLabel("0.00")
#         self._roll_lbl  = QLabel("0.00")
#         for lbl, cap in [(self._yaw_lbl, "Yaw"), (self._pitch_lbl, "Pitch"),
#                          (self._roll_lbl, "Roll")]:
#             c = QWidget()
#             cl = QVBoxLayout(c)
#             cl.setContentsMargins(4, 0, 4, 0)
#             cl.setSpacing(2)
#             cap_lbl = QLabel(cap.upper())
#             cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#             cap_lbl.setStyleSheet("font-size:8px; letter-spacing:1px;")
#             lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#             lbl.setStyleSheet("font-size:13px; font-weight:bold;")
#             cl.addWidget(cap_lbl)
#             cl.addWidget(lbl)
#             pose_vals.addWidget(c)
#         pose_l.addLayout(pose_vals)

#         bottom_row.addWidget(dist_panel, stretch=1)
#         bottom_row.addWidget(pose_panel, stretch=1)
#         col.addLayout(bottom_row)

#         return col

#     def _make_footer(self) -> QWidget:
#         w = QWidget()
#         layout = QHBoxLayout(w)
#         layout.setContentsMargins(0, 8, 0, 0)
#         l1 = QLabel("NeuralEye — Attention Tracking System")
#         l2 = QLabel("Models: gaze_model.keras · pose_model.keras · DeepFace(opencv)")
#         self._footer_ts = QLabel("—")
#         for lbl in [l1, l2, self._footer_ts]:
#             lbl.setStyleSheet("font-size:10px; letter-spacing:0.5px;")
#         layout.addWidget(l1)
#         layout.addStretch()
#         layout.addWidget(l2)
#         layout.addStretch()
#         layout.addWidget(self._footer_ts)
#         return w

#     def _mini_metric(self, label: str, value: str) -> QFrame:
#         f = QFrame()
#         f.setObjectName("miniMetric")
#         l = QVBoxLayout(f)
#         l.setContentsMargins(8, 8, 8, 8)
#         l.setSpacing(4)
#         lbl = QLabel(label)
#         lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         lbl.setStyleSheet("font-size:9px; letter-spacing:2px;")
#         val = QLabel(value)
#         val.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         val.setStyleSheet("font-size:16px; font-weight:bold;")
#         l.addWidget(lbl)
#         l.addWidget(val)
#         f.setProperty("val_lbl", val)
#         return f

#     def _make_emo_card(self, icon: str, name: str, pct: int) -> QFrame:
#         f = QFrame()
#         f.setObjectName("emoCard")
#         l = QVBoxLayout(f)
#         l.setContentsMargins(8, 10, 8, 10)
#         l.setSpacing(4)
#         icon_lbl = QLabel(icon)
#         icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         icon_lbl.setStyleSheet("font-size:22px;")
#         name_lbl = QLabel(name.upper())
#         name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         name_lbl.setStyleSheet("font-size:9px; letter-spacing:1px;")
#         pct_lbl = QLabel(f"{pct}%")
#         pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         pct_lbl.setStyleSheet("font-size:16px; font-weight:bold;")
#         bar = QProgressBar()
#         bar.setRange(0, 100)
#         bar.setValue(pct)
#         bar.setTextVisible(False)
#         bar.setFixedHeight(3)
#         l.addWidget(icon_lbl)
#         l.addWidget(name_lbl)
#         l.addWidget(pct_lbl)
#         l.addWidget(bar)
#         # Store refs
#         f.setProperty("pct_lbl", pct_lbl)
#         f.setProperty("bar", bar)
#         return f

#     def _make_model_card(self, title: str, stats: list) -> QFrame:
#         f = QFrame()
#         f.setObjectName("modelCard")
#         l = QVBoxLayout(f)
#         l.setContentsMargins(12, 12, 12, 12)
#         l.setSpacing(6)
#         title_lbl = QLabel(title.upper())
#         title_lbl.setStyleSheet("font-size:11px; font-weight:bold; letter-spacing:1px;")
#         l.addWidget(title_lbl)
#         for key, val in stats:
#             row = QWidget()
#             rl = QHBoxLayout(row)
#             rl.setContentsMargins(0, 2, 0, 2)
#             k_lbl = QLabel(key)
#             k_lbl.setStyleSheet("font-size:10px;")
#             v_lbl = QLabel(val)
#             v_lbl.setStyleSheet("font-size:10px; font-weight:bold;")
#             v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
#             rl.addWidget(k_lbl)
#             rl.addStretch()
#             rl.addWidget(v_lbl)
#             l.addWidget(row)
#         return f

#     # ── THEME APPLICATION ─────────────────────────────────────────────────────
#     def _apply_theme(self, theme_name: str):
#         global CURRENT_THEME
#         CURRENT_THEME = theme_name
#         t = THEMES[theme_name]

#         app = QApplication.instance()
#         app.setStyleSheet(make_stylesheet(theme_name))

#         # Update chart, radar, distance ring
#         self._chart.apply_theme(t)
#         self._radar.apply_theme(t)
#         self._dist_ring.apply_theme(t)

#         # Update metric cards
#         for card in [self._card_focus, self._card_distract,
#                      self._card_dist, self._card_session]:
#             card.apply_theme(t)

#         # Panel borders
#         panel_style = (
#             f"QFrame#panel{{background:{t['bg1']};border:1px solid {t['accent']}40;"
#             f"border-radius:14px;}}"
#         )
#         # Update emotion + model cards
#         emo_style = (
#             f"QFrame#emoCard{{background:{t['bg2']};border:1px solid {t['accent']}40;"
#             f"border-radius:10px;}}"
#         )
#         model_style = (
#             f"QFrame#modelCard{{background:{t['bg2']};border:1px solid {t['accent']}40;"
#             f"border-radius:10px;}}"
#         )
#         mini_style = (
#             f"QFrame#miniMetric{{background:{t['bg2']};border:1px solid {t['accent']}40;"
#             f"border-radius:8px;}}"
#         )
#         extra = panel_style + emo_style + model_style + mini_style
#         app.setStyleSheet(make_stylesheet(theme_name) + extra)

#         # Signal bar colors
#         self._status_big_lbl.setStyleSheet(
#             f"font-size:9px; letter-spacing:3px; padding-top:12px; color:{t['text3']};"
#         )
#         self._clock_lbl.setStyleSheet(f"font-size:13px; letter-spacing:1px; color:{t['text2']};")
#         self._fps_lbl.setStyleSheet(f"font-size:10px; letter-spacing:1px; color:{t['text3']};")
#         self._footer_ts.setStyleSheet(f"font-size:10px; color:{t['text3']};")
#         self._dist_status.setStyleSheet(
#             f"font-size:9px; letter-spacing:2px; padding:4px; color:{t['text3']};"
#         )
#         for lbl in [self._yaw_lbl, self._pitch_lbl, self._roll_lbl,
#                     self._dist_min_lbl, self._dist_max_lbl]:
#             lbl.setStyleSheet(f"font-size:13px; font-weight:bold; color:{t['accent3']};")
#         self._m_df.property("val_lbl").setStyleSheet(
#             f"font-size:14px; font-weight:bold; color:{t['accent3']};"
#         )

#     # ── STATE UPDATES ─────────────────────────────────────────────────────────
#     def _on_state(self, state: dict):
#         if not self._session_running:
#             return
#         t = THEMES[CURRENT_THEME]
#         focused   = state.get("focused", True)
#         gaze_conf = state.get("gaze_conf", 0.5)
#         pose_conf = state.get("pose_conf", 0.5)
#         dist      = state.get("dist", 62.0)
#         frame_n   = state.get("frame", 0)
#         df_label  = state.get("df_label", "Unknown")
#         yaw       = state.get("yaw", 0.0)
#         pitch     = state.get("pitch", 0.0)
#         roll      = state.get("roll", 0.0)
#         gaze_nx   = state.get("gaze_nx", 0.5)
#         gaze_ny   = state.get("gaze_ny", 0.5)

#         # Status
#         status_txt = "FOCUSED" if focused else "DISTRACTED"
#         status_col = t["green"] if focused else t["red"]
#         self._status_value.setText(status_txt)
#         self._status_value.setStyleSheet(
#             f"font-size:30px; font-weight:800; color:{status_col}; padding:6px;"
#         )

#         # Confidence
#         conf = gaze_conf if focused else (1 - gaze_conf)
#         pct = int(conf * 100)
#         self._conf_pct.setText(f"{pct}%")
#         self._conf_bar.setValue(pct)
#         self._conf_bar.setStyleSheet(
#             f"QProgressBar{{background:{t['bg3']};border:none;border-radius:2px;height:5px;}}"
#             f"QProgressBar::chunk{{background:{status_col};border-radius:2px;}}"
#         )

#         # Signal bars
#         gl = state.get("gaze_label", "Focused") == "Focused"
#         pl = state.get("pose_label", "Focused") == "Focused"
#         self._sig_gaze.set_value(int(gaze_conf * 100),
#                                  f"{'FOCUSED' if gl else 'DISTRACT'} {gaze_conf:.2f}", gl)
#         self._sig_pose.set_value(int(pose_conf * 100),
#                                  f"{'FOCUSED' if pl else 'DISTRACT'} {pose_conf:.2f}", pl)
#         emo_conf = state.get("df_label", "Unknown")
#         self._sig_emo.set_value(70, f"emotion {df_label.lower()[:10]}", emo_conf == "Focused")

#         # Mini metrics
#         self._m_dist.property("val_lbl").setText(f"{int(dist)}cm")
#         self._m_frame.property("val_lbl").setText(str(frame_n).zfill(4))
#         self._m_df.property("val_lbl").setText(df_label[:8].lower())

#         # Chart
#         self._chart.push(gaze_conf, pose_conf)

#         # Distance
#         if dist < self._dist_min:
#             self._dist_min = dist
#         if dist > self._dist_max:
#             self._dist_max = dist
#         self._dist_ring.set_distance(dist)
#         if dist > 90:
#             dstatus = "TOO FAR"
#         elif dist < 30:
#             dstatus = "TOO CLOSE"
#         else:
#             dstatus = "OPTIMAL RANGE"
#         self._dist_status.setText(dstatus)
#         self._dist_min_lbl.setText(f"{int(self._dist_min)}cm")
#         self._dist_max_lbl.setText(f"{int(self._dist_max)}cm")

#         # Head pose
#         self._radar.set_pose(yaw, pitch, roll)
#         self._yaw_lbl.setText(f"{yaw:.2f}")
#         self._pitch_lbl.setText(f"{pitch:.2f}")
#         self._roll_lbl.setText(f"{roll:.2f}")

#         # Focus rate
#         self._focus_history.append(1 if focused else 0)
#         self._focus_history = self._focus_history[-300:]
#         if self._focus_history:
#             rate = int(sum(self._focus_history) / len(self._focus_history) * 100)
#             self._focus_rate = rate
#             self._card_focus.set_value(f"{rate}%")
#             self._card_focus.set_color(t["green"] if rate > 75 else t["yellow"])

#         if not focused:
#             self._distract_cnt += 1
#         self._card_distract.set_value(str(self._distract_cnt))
#         self._card_dist.set_value(f"{int(dist)} cm")

#         # Gaze overlay
#         self._gaze_overlay.update_gaze(gaze_nx, gaze_ny)

#     def _on_frame(self, frame: np.ndarray):
#         if self._cam_active:
#             state = self._worker._engine.state if (
#                 ENGINE_AVAILABLE and self._worker._engine
#             ) else {}
#             self._cam_widget.update_frame(
#                 frame,
#                 state.get("gaze_nx", 0.5) if state else 0.5,
#                 state.get("gaze_ny", 0.5) if state else 0.5
#             )

#     # ── CONTROLS ──────────────────────────────────────────────────────────────
#     def _toggle_camera(self):
#         self._cam_active = not self._cam_active
#         if self._cam_active:
#             self._worker.resume()
#             self._btn_cam.setText("⏸  Pause Camera")
#             self._append_log("Camera feed resumed.", "ok")
#         else:
#             self._worker.pause()
#             self._btn_cam.setText("▶  Start Camera")
#             self._append_log("Camera feed paused.", "warn")

#     def _export_log(self):
#         txt = self._log_text.toPlainText()
#         path = os.path.join(os.path.dirname(__file__), "attention_log.txt")
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(txt)
#         self._append_log(f"Log exported to {path}", "ok")

#     def _stop_session(self):
#         self._session_running = False
#         self._worker.pause()
#         self._sys_dot.setStyleSheet("font-size:10px; color:#ef4444;")
#         self._sys_status.setText("SESSION STOPPED")
#         self._append_log("Session stopped by user.", "err")

#     def _force_foreground(self):
#         self.showNormal()
#         self.raise_()
#         self.activateWindow()

#     # ── TIMERS ────────────────────────────────────────────────────────────────
#     def _tick_clock(self):
#         from datetime import datetime
#         ts = datetime.now().strftime("%H:%M:%S")
#         self._clock_lbl.setText(ts)
#         self._footer_ts.setText(f"Last update: {ts}")

#     def _tick_session(self):
#         if not self._session_running:
#             return
#         self._session_sec += 1
#         m = str(self._session_sec // 60).zfill(2)
#         s = str(self._session_sec % 60).zfill(2)
#         self._card_session.set_value(f"{m}:{s}")

#     # ── LOGGING ───────────────────────────────────────────────────────────────
#     def _append_log(self, msg: str, typ: str = "info"):
#         m = self._session_sec // 60
#         s = self._session_sec % 60
#         ts = f"{m:02d}:{s:02d}"
#         colors = {"ok": "#22c55e", "warn": "#f59e0b", "err": "#ef4444", "info": "#9d5cf6"}
#         col = colors.get(typ, "#9d5cf6")
#         html = f'<span style="color:#6b4d8c">{ts} </span><span style="color:{col}">{msg}</span><br>'
#         self._log_text.insertHtml(html)
#         self._log_text.verticalScrollBar().setValue(
#             self._log_text.verticalScrollBar().maximum()
#         )

#     # ── CLOSE ─────────────────────────────────────────────────────────────────
#     def closeEvent(self, event):
#         self._enforcer.stop()
#         self._worker.stop()
#         self._gaze_overlay.close()
#         self._countdown_overlay.close()
#         event.accept()


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════════
# def main():
#     app = QApplication(sys.argv)
#     app.setApplicationName("NeuralEye")
#     app.setApplicationDisplayName("NeuralEye — Attention Tracking System")

#     # Try to load Space Mono font
#     try:
#         QFontDatabase.addApplicationFont("SpaceMono-Regular.ttf")
#         QFontDatabase.addApplicationFont("SpaceMono-Bold.ttf")
#     except Exception:
#         pass

#     app.setStyleSheet(make_stylesheet("Dark Purple"))

#     window = MainWindow()
#     window.show()
#     window._append_log("System initialized. Models loading…", "info")
#     window._append_log("Camera feed starting…", "info")

#     sys.exit(app.exec())


# if __name__ == "__main__":
#     main()