"""
DOG-GS desktop app (PySide6). Drop a GeoSplat project folder or a georeferenced Gaussian .PLY
(+ optional RTK checkpoint .CSV and reference orthophoto .TIF) into the window and press Start.

    python -m doggs.gui [MODEL.PLY | GEOSPLAT_FOLDER] [CHECKPOINTS.CSV] [REFERENCE.TIF]
"""
from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from . import __version__, about
from .geosplat import autodetect_crs_json, load_geosplat_crs
from .pipeline import (STAGES, Job, Pipeline, default_output, default_stem, environment, find_model, load_settings,
                       open_path, save_settings)

ROOT = Path(__file__).resolve().parents[2]
ACC = "#ff6a13"
MUTED = "#8f8f8f"
OK, BAD = "#53c27a", "#ff5b5b"
QSS = f"""
QWidget {{ background:#18181a; color:#e8e8e8; font-family:'Segoe UI'; font-size:10pt; }}
QMenuBar {{ background:#18181a; color:#cfcfcf; padding:2px 6px; }}
QMenuBar::item:selected {{ background:#2c2c2e; border-radius:4px; }}
QMenu {{ background:#232325; border:1px solid #333; padding:4px; }}
QMenu::item {{ padding:6px 22px; border-radius:4px; }}
QMenu::item:selected {{ background:#333336; }}
QFrame#card {{ background:#222224; border:1px solid #2e2e30; border-radius:10px; }}
QLabel#title {{ font-size:17pt; font-weight:600; letter-spacing:0.3px; background:transparent; }}
QLabel#badge {{ background:#2e2e30; color:#bdbdbd; border-radius:9px; padding:2px 9px; font-size:8.5pt; }}
QLabel#key {{ color:{MUTED}; background:transparent; }}
QLabel#section {{ color:{MUTED}; font-size:8.5pt; font-weight:600; letter-spacing:1px; background:transparent; }}
QLabel#footer {{ color:#6f6f6f; font-size:8.5pt; background:transparent; }}
QLabel#preview {{ background:#141416; border:1px solid #2a2a2c; border-radius:8px; color:#5f5f5f; }}
QFrame#drop {{ background:#1d1d1f; border:2px dashed #3d3d40; border-radius:12px; }}
QFrame#drop[hover="true"] {{ border-color:{ACC}; background:#261d15; }}
QPushButton {{ background:#2f2f32; border:0; border-radius:6px; padding:7px 16px; }}
QPushButton:hover {{ background:#3a3a3d; }}
QPushButton:disabled {{ color:#5f5f5f; background:#252527; }}
QPushButton#primary {{ background:{ACC}; color:#111; font-weight:600; }}
QPushButton#primary:hover {{ background:#ff7d33; }}
QPushButton#primary:disabled {{ background:#4d3424; color:#8a8a8a; }}
QPushButton#link {{ background:transparent; color:#9fb4ff; padding:0 4px; font-size:8.5pt; }}
QPushButton#link:hover {{ color:#c7d3ff; text-decoration:underline; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:#141416; border:1px solid #353538; border-radius:5px;
    padding:4px 6px; }}
QDoubleSpinBox:disabled {{ color:#5f5f5f; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color:{ACC}; }}
QCheckBox {{ background:transparent; spacing:6px; }}
QPlainTextEdit {{ background:#111113; border:1px solid #2a2a2c; border-radius:8px; font-family:Consolas;
    font-size:9pt; color:#bdbdbd; }}
QTextBrowser {{ background:#1d1d1f; border:0; }}
QProgressBar {{ background:#141416; border:0; border-radius:3px; max-height:6px; }}
QProgressBar::chunk {{ background:{ACC}; border-radius:3px; }}
QToolTip {{ background:#2a2a2c; color:#e8e8e8; border:1px solid #3a3a3c; padding:4px; }}
"""
ICON = {"pending": ("○", "#6a6a6a"), "running": ("◐", ACC), "done": ("✓", OK), "skipped": ("–", "#7a7a7a"),
        "failed": ("✗", BAD), "cancelled": ("■", "#999")}
NO_CHECKPOINTS = "None (optional; RTK/GNSS points with ellipsoidal heights to check the DSM)"
NO_REFERENCE = "None (optional; photogrammetric orthophoto GeoTIFF for a relative comparison)"
COLOURS = [("sh", "SH at nadir (view-dependent colour, d = 0,0,−1)"), ("dc", "DC (view-independent colour)")]


class Worker(QObject):
    log = Signal(str)
    stage = Signal(str, str, str)
    progress = Signal(float)
    finished = Signal(bool)

    def __init__(self, job):
        super().__init__()
        self.pl = Pipeline(job, on_log=self.log.emit, on_stage=lambda k, s, i="": self.stage.emit(k, s, i),
                           on_progress=self.progress.emit)

    def run(self):
        try:
            ok = self.pl.run()
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"ERROR: {e}")
            ok = False
        self.finished.emit(ok)


class BuildWorker(QObject):
    finished = Signal(bool, str)

    def run(self):
        try:
            from .kernel import build_c
            p = build_c(verbose=False)
            self.finished.emit(True, str(p))
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e))


class DropZone(QFrame):
    def __init__(self, on_files):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.on_files = on_files
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 22, 20, 22)
        lay.setSpacing(8)
        t = QLabel("Drop a GeoSplat project folder or a Gaussian .PLY here")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-size:13pt; font-weight:600; background:transparent;")
        s = QLabel("The georeferenced UTM model (…_hcorr.ply) is selected automatically from a GeoSplat folder. "
                   "Optionally add an RTK checkpoint .CSV and a reference orthophoto .TIF for validation.")
        s.setAlignment(Qt.AlignCenter)
        s.setWordWrap(True)
        s.setStyleSheet(f"color:{MUTED}; background:transparent;")
        row = QHBoxLayout()
        row.addStretch()
        b1 = QPushButton("Browse files…")
        b1.clicked.connect(self.pick)
        b2 = QPushButton("Choose GeoSplat folder…")
        b2.clicked.connect(self.pick_folder)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch()
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addLayout(row)

    def pick(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Gaussian PLY, checkpoint CSV and/or reference orthophoto", "",
            "Model / checkpoints / reference (*.ply *.PLY *.csv *.CSV *.tif *.tiff *.TIF *.TIFF)")
        if files:
            self.on_files(files)

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "GeoSplat project folder")
        if d:
            self.on_files([d])

    def _hover(self, v):
        self.setProperty("hover", v)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        self.on_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


class AboutDialog(QDialog):
    """Authors, affiliations, licence, acknowledgements and citation."""

    def __init__(self, parent=None, focus_cite=False):
        super().__init__(parent)
        self.setWindowTitle(f"About {about.NAME}")
        self.resize(660, 620)
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        txt = QTextBrowser()
        txt.setOpenExternalLinks(True)
        txt.setHtml(self._html())
        v.addWidget(txt, 1)
        row = QHBoxLayout()
        for label, fn in (("Copy citation (APA)", about.citation_apa), ("Copy BibTeX", about.citation_bibtex)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, f=fn, btn=b: self._copy(f(), btn))
            row.addWidget(b)
        row.addStretch()
        close = QPushButton("Close")
        close.setObjectName("primary")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)
        if focus_cite:
            txt.scrollToAnchor("cite")

    @staticmethod
    def _copy(text, btn):
        QGuiApplication.clipboard().setText(text)
        old = btn.text()
        btn.setText("Copied ✓")
        QTimer.singleShot(1500, lambda: btn.setText(old))

    @staticmethod
    def _html():
        authors = ", ".join(f"<b>{n}</b><sup>{a}{',*' if c else ''}</sup>" for n, a, c in about.AUTHORS)
        affs = "<br>".join(f"<sup>{k}</sup> {v}" for k, v in about.AFFILIATIONS.items())
        ack = "".join(f"<li><b>{n}</b> — {who} <span style='color:{MUTED}'>({lic})</span></li>"
                      for n, who, lic in about.ACKNOWLEDGEMENTS)
        return f"""
        <div style="font-family:'Segoe UI'; color:#e8e8e8;">
        <p style="font-size:18pt; font-weight:600; margin:0">{about.NAME}
           <span style="font-size:10pt; color:{MUTED}; font-weight:400">version {__version__}</span></p>
        <p style="margin-top:2px">{about.FULL_NAME}</p>
        <p style="color:{MUTED}; margin-top:0">{about.TAGLINE}</p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px">AUTHORS</p>
        <p style="margin-top:0">{authors}</p>
        <p style="font-size:9pt; color:#c8c8c8">{affs}<br><sup>*</sup> Corresponding author:
           <a style="color:#9fb4ff" href="mailto:{about.CORRESPONDENCE}">{about.CORRESPONDENCE}</a></p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px">METHOD</p>
        <p style="margin-top:0; font-size:9.5pt">Each Gaussian is projected orthographically onto the horizontal plane
           (Σ<sub>2D</sub> = [Σ]<sub>EN</sub>), sorted by decreasing height and alpha-composited front-to-back:
           C = Σ c<sub>i</sub> α<sub>i</sub> T<sub>i</sub>, T<sub>i</sub> = Π<sub>j&lt;i</sub>(1 − α<sub>j</sub>).
           The DSM is the height at which the transmittance first drops below 0.5.</p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px"><a name="cite"></a>HOW TO CITE</p>
        <p style="margin-top:0; font-size:9.5pt">{about.citation_apa()}</p>
        <p style="font-size:9pt; color:#c8c8c8">Please also cite 3D Gaussian Splatting (Kerbl et al., 2023) and, when the
           model was produced with it, GeoSplat.</p>
        <p style="color:{MUTED}; font-size:8.5pt; letter-spacing:1px; margin-bottom:4px">LICENCE &amp; ACKNOWLEDGEMENTS</p>
        <p style="margin-top:0; font-size:9.5pt">DOG-GS is released under the {about.LICENSE} licence. It builds on:</p>
        <ul style="font-size:9.5pt">{ack}</ul>
        <p style="font-size:9.5pt">Source code: <a style="color:#9fb4ff" href="{about.REPO_URL}">{about.REPO_URL}</a></p>
        </div>"""


def _spin(lo, hi, step, val, suffix="", decimals=2, tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setToolTip(tip)
    return s


class Main(QMainWindow):
    def __init__(self, argv_files):
        super().__init__()
        self.setWindowTitle(f"{about.NAME} {__version__}")
        self.resize(1080, 900)
        self.settings = load_settings()
        self.ply = self.root = None
        self.out_dir = None
        self.checkpoints = self.reference = None
        self.ortho_path = None
        self.thread = self.worker = None
        self.t_start = None
        self.cur = None
        self._menus()

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(20, 12, 20, 10)
        v.setSpacing(12)

        # header ---------------------------------------------------------------
        head = QHBoxLayout()
        mark = QLabel("◆")
        mark.setStyleSheet(f"color:{ACC}; font-size:17pt; background:transparent;")
        t = QLabel(about.NAME)
        t.setObjectName("title")
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("badge")
        head.addWidget(mark)
        head.addWidget(t)
        head.addSpacing(4)
        head.addWidget(ver, alignment=Qt.AlignVCenter)
        head.addStretch()
        self.pre = QLabel()
        self.pre.setStyleSheet("background:transparent;")
        self.pre.setToolTip("Rendering backends and validation tools detected on this computer")
        head.addWidget(self.pre)
        head.addSpacing(10)
        self.bbuild = QPushButton("Build C backend…")
        self.bbuild.setToolTip("Compile the optional C/OpenMP renderer (needs gcc/clang; not needed when Numba is installed)")
        self.bbuild.clicked.connect(self.build_c)
        head.addWidget(self.bbuild)
        v.addLayout(head)

        self.drop = DropZone(self.set_files)
        v.addWidget(self.drop)

        # project card -----------------------------------------------------------
        card = QFrame()
        card.setObjectName("card")
        g = QGridLayout(card)
        g.setContentsMargins(18, 14, 18, 14)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(9)
        sec = QLabel("PROJECT")
        sec.setObjectName("section")
        g.addWidget(sec, 0, 0, 1, 7)
        self.lm, self.lg, self.lo = QLabel("–"), QLabel("–"), QLabel("–")
        self.lc, self.lr = QLabel(NO_CHECKPOINTS), QLabel(NO_REFERENCE)
        for lab in (self.lm, self.lg, self.lo, self.lc, self.lr):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setWordWrap(True)
        rows = (("Gaussian model", self.lm, None), ("Georeferencing", self.lg, None),
                ("Output folder", self.lo, ("Change…", self.pick_output)),
                ("RTK checkpoints", self.lc, ("Browse…", self.pick_checkpoints)),
                ("Reference ortho", self.lr, ("Browse…", self.pick_reference)))
        for r, (name, w, btn) in enumerate(rows, 1):
            k = self._key(name)
            g.addWidget(k, r, 0)
            g.addWidget(w, r, 1, 1, 5)
            if btn:
                b = QPushButton(btn[0])
                b.clicked.connect(btn[1])
                g.addWidget(b, r, 6)
        self.lc.setToolTip("CSV with id, E/easting, N/northing, h (ellipsoidal) and optional class/role.\n"
                           "Points whose class contains 'skip' or 'excl' are ignored.")
        self.lr.setToolTip("Orthophoto GeoTIFF in the same UTM zone or in EPSG:4326 (palette TIFFs are supported).\n"
                           "Needs OpenCV.")

        sec2 = QLabel("SETTINGS")
        sec2.setObjectName("section")
        g.addWidget(sec2, 6, 0, 1, 7)
        self.gsd = _spin(0.01, 2.0, 0.01, float(self.settings.get("gsd", 0.06)), " m", 3,
                         "Ground sampling distance of the orthophoto and DSM")
        self.col = QComboBox()
        for key, label in COLOURS:
            self.col.addItem(label, key)
        self.col.setCurrentIndex(0 if self.settings.get("colour", "sh") == "sh" else 1)
        self.be = QComboBox()
        self.be.addItems(["auto", "numba", "c", "python"])
        self.be.setCurrentText(self.settings.get("backend", "auto"))
        self.be.setToolTip("auto: Numba if installed, otherwise C, otherwise pure Python (slow)")
        for c, (name, w) in enumerate((("GSD", self.gsd), ("Colour", self.col), ("Backend", self.be))):
            g.addWidget(self._key(name), 7, c * 2)
            g.addWidget(w, 7, c * 2 + 1)

        self.auto = QCheckBox("Automatic")
        self.auto.setChecked(bool(self.settings.get("auto_z", True)))
        self.auto.setToolTip("zmin = ground − 10 m; zmax = ground + 0.6 × (camera height − ground)\n"
                             "ground = 5th percentile of Gaussian heights; camera height from camera_trajectory*.csv")
        self.zmin = _spin(-500, 9000, 1, float(self.settings.get("zmin", 20.0)), " m", 2, "Drop Gaussians below this height")
        self.zmax = _spin(-500, 9000, 1, float(self.settings.get("zmax", 110.0)), " m", 2,
                          "Drop Gaussians above this height (floaters near the camera path)")
        self.auto.toggled.connect(lambda on: (self.zmin.setEnabled(not on), self.zmax.setEnabled(not on)))
        self.zmin.setEnabled(not self.auto.isChecked())
        self.zmax.setEnabled(not self.auto.isChecked())
        self.mask = QCheckBox("Clip to valid region")
        self.mask.setChecked(bool(self.settings.get("mask_region", False)))
        self.mask.setToolTip("Set alpha = 0 outside the GeoSplat valid region (circle)")
        g.addWidget(self._key("Height window"), 8, 0)
        hw = QHBoxLayout()
        hw.addWidget(self.auto)
        hw.addSpacing(10)
        hw.addWidget(self._key("min"))
        hw.addWidget(self.zmin)
        hw.addWidget(self._key("max"))
        hw.addWidget(self.zmax)
        hw.addSpacing(16)
        hw.addWidget(self.mask)
        hw.addStretch()
        g.addLayout(hw, 8, 1, 1, 6)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setColumnStretch(5, 1)
        v.addWidget(card)

        # stages + preview ----------------------------------------------------------
        stc = QFrame()
        stc.setObjectName("card")
        outer = QHBoxLayout(stc)
        outer.setContentsMargins(18, 12, 18, 12)
        sgw = QWidget()
        sgw.setStyleSheet("background:transparent;")
        sg = QGridLayout(sgw)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setVerticalSpacing(5)
        sec3 = QLabel("PROCESSING")
        sec3.setObjectName("section")
        sg.addWidget(sec3, 0, 0, 1, 3)
        self.st = {}
        for r, (key, label) in enumerate(STAGES):
            ic = QLabel()
            ic.setFixedWidth(22)
            nm = QLabel(f"{r + 1}.  {label}")
            nm.setStyleSheet("background:transparent;")
            inf = QLabel("")
            inf.setStyleSheet(f"color:{MUTED}; background:transparent;")
            sg.addWidget(ic, r + 1, 0)
            sg.addWidget(nm, r + 1, 1)
            sg.addWidget(inf, r + 1, 2)
            self.st[key] = (ic, inf)
            self.set_stage(key, "pending")
        sg.setColumnStretch(2, 1)
        sg.setRowStretch(len(STAGES) + 1, 1)
        outer.addWidget(sgw, 1)
        self.preview = QLabel("orthophoto preview")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(230, 230)
        outer.addWidget(self.preview)
        v.addWidget(stc)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.now = QLabel("Ready.")
        self.now.setStyleSheet(f"color:{MUTED};")
        v.addWidget(self.bar)
        v.addWidget(self.now)

        self.logw = QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(5000)
        self.logw.setPlaceholderText("Processing log")
        v.addWidget(self.logw, 1)

        btns = QHBoxLayout()
        self.bstart = QPushButton("Start")
        self.bstart.setObjectName("primary")
        self.bstart.setMinimumWidth(150)
        self.bstart.clicked.connect(self.start)
        self.bstop = QPushButton("Cancel")
        self.bstop.clicked.connect(self.cancel)
        self.bfold = QPushButton("Open output folder")
        self.bfold.clicked.connect(lambda: open_path(self.out_dir))
        self.bortho = QPushButton("Open orthophoto")
        self.bortho.clicked.connect(lambda: open_path(self.ortho_path))
        btns.addWidget(self.bstart)
        btns.addWidget(self.bstop)
        btns.addStretch()
        btns.addWidget(self.bfold)
        btns.addWidget(self.bortho)
        v.addLayout(btns)

        # footer: permanent author credit ---------------------------------------------
        foot = QHBoxLayout()
        cred = QLabel(about.credit_line())
        cred.setObjectName("footer")
        cred.setToolTip("\n".join(f"{n} — {about.AFFILIATIONS[a]}" for n, a, _ in about.AUTHORS))
        foot.addWidget(cred)
        foot.addStretch()
        for label, cite in (("How to cite", True), ("About", False)):
            b = QPushButton(label)
            b.setObjectName("link")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, c=cite: self.show_about(c))
            foot.addWidget(b)
        v.addLayout(foot)

        self.refresh_env()
        self.update_buttons()
        if argv_files:
            self.set_files(argv_files)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    @staticmethod
    def _key(text):
        k = QLabel(text)
        k.setObjectName("key")
        return k

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for label, fn in (("Open Gaussian model / files…", lambda: self.drop.pick()),
                          ("Open GeoSplat project folder…", lambda: self.drop.pick_folder()),
                          ("Open RTK checkpoints…", self.pick_checkpoints),
                          ("Open reference orthophoto…", self.pick_reference),
                          ("Change output folder…", self.pick_output)):
            a = QAction(label, self)
            a.triggered.connect(fn)
            f.addAction(a)
        f.addSeparator()
        q = QAction("Exit", self)
        q.triggered.connect(self.close)
        f.addAction(q)
        h = mb.addMenu("&Help")
        items = (("User guide (README)", lambda: open_path(ROOT / "README.md")),
                 ("Project website", lambda: webbrowser.open(about.REPO_URL)),
                 (None, None),
                 ("How to cite DOG-GS…", lambda: self.show_about(True)),
                 ("About DOG-GS…", lambda: self.show_about(False)))
        for label, fn in items:
            if label is None:
                h.addSeparator()
                continue
            a = QAction(label, self)
            a.triggered.connect(fn)
            h.addAction(a)

    def show_about(self, cite=False):
        AboutDialog(self, cite).exec()

    # ------------------------------------------------------------------ state
    def refresh_env(self):
        e = environment()
        self.env = e
        part = lambda ok, name: f"<span style='color:{OK if ok else BAD}'>{'✓' if ok else '✗'} {name}</span>"  # noqa: E731
        self.pre.setText("&nbsp;&nbsp;".join([part(e["numba"], "Numba"), part(e["c"], "C backend"),
                                              part(e["opencv"], "OpenCV")]))
        self.bbuild.setVisible(not e["c"])

    def set_files(self, files):
        for f in files:
            p = Path(f)
            ext = p.suffix.lower()
            if p.is_dir() or ext == ".ply":
                ply, root = find_model(p)
                if ply is None:
                    QMessageBox.warning(self, about.NAME, f"No Gaussian PLY found in:\n{p}")
                    continue
                self.ply, self.root = str(ply), root
                self.out_dir = default_output(ply, root)
                self.ortho_path = None
                self.preview.setPixmap(QPixmap())
                self.preview.setText("orthophoto preview")
                self.describe_model()
            elif ext == ".csv":
                self.checkpoints = str(p.resolve())
                self.lc.setText(self.checkpoints)
            elif ext in (".tif", ".tiff"):
                self.reference = str(p.resolve())
                self.lr.setText(self.reference)
        if self.root is not None and not self.checkpoints:
            cand = sorted(Path(self.root).glob("10_Vertical_Correction/checkpoints_dz.csv")) or \
                sorted(Path(self.root).parent.glob("*checkpoint*.csv"))
            if cand:
                self.checkpoints = str(cand[0])
                self.lc.setText(self.checkpoints + "   (found next to the project)")
        self.lo.setText(self.out_dir or "–")
        for key, _ in STAGES:
            self.set_stage(key, "pending")
        self.update_buttons()

    def describe_model(self):
        self.lm.setText(self.ply)
        js = autodetect_crs_json(self.ply)
        if js is None:
            self.lg.setText("<span style='color:%s'>No GeoSplat gaussians_crs*.json next to the PLY — use the CLI "
                            "with --epsg/--offset for other models</span>" % BAD)
            return
        try:
            gf = load_geosplat_crs(js, Path(self.ply).name)
            extra = ""
            if gf.region_radius:
                extra = f" · valid region radius {gf.region_radius:.0f} m"
            hr = "vertical datum corrected" if "hcorr" in Path(self.ply).name else "heights not datum-corrected"
            self.lg.setText(f"EPSG:{gf.epsg} · offset {gf.offset[0]:.0f}, {gf.offset[1]:.0f} · {hr}{extra}   "
                            f"<span style='color:{MUTED}'>({js.name})</span>")
        except Exception as e:  # noqa: BLE001
            self.lg.setText(f"<span style='color:{BAD}'>Cannot read {js.name}: {e}</span>")

    def pick_checkpoints(self):
        f, _ = QFileDialog.getOpenFileName(self, "RTK checkpoint CSV", str(Path(self.ply).parent if self.ply else ""),
                                           "CSV (*.csv)")
        if f:
            self.set_files([f])

    def pick_reference(self):
        f, _ = QFileDialog.getOpenFileName(self, "Reference orthophoto", "", "GeoTIFF (*.tif *.tiff)")
        if f:
            self.set_files([f])

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir or "")
        if d:
            self.out_dir = d
            self.lo.setText(d)
            self.update_buttons()

    def set_stage(self, key, status, info=""):
        ic, inf = self.st[key]
        sym, col = ICON.get(status, ("?", "#999"))
        ic.setText(sym)
        ic.setStyleSheet(f"color:{col}; font-size:13pt; background:transparent;")
        inf.setText(info)
        if status == "running":
            self.cur = (key, time.time())
            self.now.setText(dict(STAGES)[key] + "…")

    def update_buttons(self):
        running = self.thread is not None
        self.bstart.setEnabled(bool(self.ply) and not running)
        self.bstop.setEnabled(running)
        self.bfold.setEnabled(bool(self.out_dir) and os.path.isdir(self.out_dir or "") and not running)
        self.bortho.setEnabled(bool(self.ortho_path) and os.path.exists(self.ortho_path or "") and not running)

    def tick(self):
        if self.thread and self.cur:
            k, t0 = self.cur
            el = time.time() - t0
            if k == "render":
                self.st[k][1].setText(f"running {int(el // 60)}:{int(el % 60):02d}")

    # ------------------------------------------------------------------ actions
    def start(self):
        auto = self.auto.isChecked()
        self.settings.update({"gsd": self.gsd.value(), "colour": self.col.currentData(), "backend": self.be.currentText(),
                              "auto_z": auto, "zmin": self.zmin.value(), "zmax": self.zmax.value(),
                              "mask_region": self.mask.isChecked()})
        save_settings(self.settings)
        if self.be.currentText() == "auto" and not (self.env["numba"] or self.env["c"]):
            r = QMessageBox.question(self, about.NAME, "Neither Numba nor the C backend is available, so the pure-Python "
                                     "renderer would be used (very slow for real models).\n\n"
                                     "Install Numba (pip install numba) or build the C backend first. Continue anyway?")
            if r != QMessageBox.Yes:
                return
        if self.reference and not self.env["opencv"]:
            QMessageBox.warning(self, about.NAME, "The comparison with a reference orthophoto needs OpenCV "
                                                  "(pip install opencv-python). That stage will be skipped.")
        stem = default_stem(self.ply, self.root)
        job = Job(ply=self.ply, out_dir=self.out_dir, stem=stem, gsd=self.gsd.value(), colour=self.col.currentData(),
                  zmin="auto" if auto else self.zmin.value(), zmax="auto" if auto else self.zmax.value(),
                  backend=self.be.currentText(), checkpoints=self.checkpoints,
                  reference=self.reference if self.env["opencv"] else None, mask_region=self.mask.isChecked())
        self.ortho_path = str(Path(self.out_dir) / f"{stem}_ortho.tif")
        self.logw.clear()
        self.bar.setValue(0)
        for key, _ in STAGES:
            self.set_stage(key, "pending")
        self.thread = QThread()
        self.worker = Worker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.on_log)
        self.worker.stage.connect(self.set_stage)
        self.worker.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        self.worker.finished.connect(self.on_done)
        self.thread.start()
        self.t_start = time.time()
        self.now.setText("Running…")
        self.update_buttons()

    def on_log(self, line):
        if line.strip():
            self.logw.appendPlainText(line)

    def on_done(self, ok):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        el = (time.time() - self.t_start) if self.t_start else 0
        self.now.setText(("Finished" if ok else "Stopped") + f" · {el:.0f} s")
        if ok:
            self.show_preview()
        self.update_buttons()

    def show_preview(self):
        try:
            import tifffile
            with tifffile.TiffFile(self.ortho_path) as t:
                page = t.pages[0]
                h, w = page.shape[:2]
                a = page.asarray()
            step = max(1, max(h, w) // 460)
            a = np.ascontiguousarray(a[::step, ::step, :4])
            img = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGBA8888).copy()
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                                 Qt.SmoothTransformation))
        except Exception as e:  # noqa: BLE001
            self.preview.setText(f"preview unavailable\n{e}")

    def cancel(self):
        if self.worker:
            self.worker.pl.stop()
            self.now.setText("Cancelling…")

    def build_c(self):
        self.bbuild.setEnabled(False)
        self.bbuild.setText("Building…")
        self.bthread = QThread()
        self.bworker = BuildWorker()
        self.bworker.moveToThread(self.bthread)
        self.bthread.started.connect(self.bworker.run)

        def done(ok, msg):
            self.bthread.quit()
            self.bthread.wait()
            self.bbuild.setEnabled(True)
            self.bbuild.setText("Build C backend…")
            self.refresh_env()
            (QMessageBox.information if ok else QMessageBox.warning)(
                self, about.NAME, ("C backend built:\n" if ok else "C backend could not be built:\n") + msg)

        self.bworker.finished.connect(done)
        self.bthread.start()

    def closeEvent(self, e):
        if self.thread:
            if QMessageBox.question(self, about.NAME, "Processing is still running. Stop it and exit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self.cancel()
            self.thread.quit()
            self.thread.wait(5000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(about.NAME)
    app.setApplicationVersion(__version__)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    w = Main([a for a in sys.argv[1:] if os.path.exists(a)])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
