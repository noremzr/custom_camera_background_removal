#!/usr/bin/env python3
"""
camfundo - remoção de fundo da webcam com IA (alternativa ao Nvidia Broadcast).

Pipeline: webcam -> Robust Video Matting (ONNX) na GPU via DirectML
          -> composição (blur / imagem / cor) -> câmera virtual do OBS.

Uso rápido:
    python camfundo.py --preview
    python camfundo.py --mode image --image fundo.jpg
Teclas na janela de preview: b = blur, c = cor, i = imagem, n = sem efeito,
  + / - = BPM de teste, h = mostra/esconde o coração, q/Esc = sair
"""
import argparse
from collections import deque
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

BASE = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
MODEL_FILES = {
    "mobilenet": "rvm_mobilenetv3_fp32.onnx",  # rápido, boa qualidade
    "resnet50": "rvm_resnet50_fp32.onnx",      # mais pesado, bordas melhores
}


def ensure_model(name):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, MODEL_FILES[name])
    if not os.path.exists(path):
        print(f"Baixando modelo {MODEL_FILES[name]} (só na primeira vez)...")
        urllib.request.urlretrieve(BASE + MODEL_FILES[name], path + ".part")
        os.replace(path + ".part", path)
    return path


class Matter:
    """Executa o RVM mantendo o estado recorrente (consistência temporal)."""

    def __init__(self, model_path, infer_width=640, ratio=None, device="dml",
                 edge=0.08, shrink=0, threads=0):
        available = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider")
                         if p in available] + ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        if "DmlExecutionProvider" in providers:
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(model_path, opts, providers=providers)
        print("Provedor em uso:", self.sess.get_providers()[0])
        self.infer_width = infer_width
        self.ratio = ratio
        self.edge = edge
        self.kernel = (np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8)
                       if shrink > 0 else None)
        self.reset()

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    def alpha(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        iw = max(16, self.infer_width // 16 * 16)
        ih = max(16, int(round(self.infer_width * h / w)) // 16 * 16)
        ratio = self.ratio or min(1.0, 256 / min(iw, ih))

        small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        src = np.ascontiguousarray(
            (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

        out = self.sess.run(None, {
            "src": src,
            "r1i": self.rec[0], "r2i": self.rec[1],
            "r3i": self.rec[2], "r4i": self.rec[3],
            "downsample_ratio": np.array([ratio], np.float32),
        })
        self.rec = list(out[2:6])
        a = out[1][0, 0]

        # curva que remove "névoa" nas bordas e firma o interior (em baixa resolução)
        lo, hi = self.edge, 1.0 - self.edge
        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        a8 = (a * 255 + 0.5).astype(np.uint8)
        a8 = cv2.resize(a8, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.kernel is not None:
            a8 = cv2.erode(a8, self.kernel)
        return a8


class AsyncMatter:
    """Roda a IA em outra thread; o loop principal usa sempre a máscara mais recente."""

    def __init__(self, matter, ia_fps=0):
        self.matter = matter
        self.min_interval = 1.0 / ia_fps if ia_fps > 0 else 0.0
        self.pending = None
        self.mask = None
        self.mask_ref = None
        self.mask_fid = -1
        self.count = 0
        self.running = True
        self.lock = threading.Lock()
        self.has_work = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            if not self.has_work.wait(0.5):
                continue
            self.has_work.clear()
            with self.lock:
                job, self.pending = self.pending, None
            if job is None:
                continue
            f, ref, fid = job
            t0 = time.perf_counter()
            a = self.matter.alpha(f)
            with self.lock:
                self.mask, self.mask_ref, self.mask_fid = a, ref, fid
                self.count += 1
            self.ready.set()
            resto = self.min_interval - (time.perf_counter() - t0)
            if resto > 0:
                time.sleep(resto)

    def submit(self, frame, ref=None, fid=0):
        with self.lock:
            self.pending = (frame, ref, fid)
        self.has_work.set()

    def latest(self):
        self.ready.wait(10)
        with self.lock:
            return self.mask, self.mask_ref, self.mask_fid

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


_grid = {}


def small_gray(frame, fw=256):
    h, w = frame.shape[:2]
    fh = max(2, int(round(fw * h / w)))
    small = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def warp_mask(mask, ref_small, cur_small):
    """Move a máscara antiga para a posição atual usando fluxo óptico (compensa movimento)."""
    h, w = mask.shape
    fh, fw = cur_small.shape
    if _grid.get("shape") != (h, w):
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        _grid.update(shape=(h, w), gx=gx, gy=gy)
    flow = cv2.calcOpticalFlowFarneback(cur_small, ref_small, None, 0.5, 3, 15, 2, 5, 1.1, 0)
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    map_x = _grid["gx"] + flow[..., 0] * (w / fw)
    map_y = _grid["gy"] + flow[..., 1] * (h / fh)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compose(frame, a8, bg):
    """Mistura frame e fundo usando a máscara uint8 (aritmética inteira, rápida)."""
    a3 = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)
    fg = cv2.multiply(frame, a3, scale=1 / 255.0)
    bk = cv2.multiply(bg, cv2.bitwise_not(a3), scale=1 / 255.0)
    return cv2.add(fg, bk)


class Grabber:
    """Lê a webcam em outra thread, sempre entregando o frame mais recente."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ts = 0.0
        self.frame_ts = 0.0   # instante (perf_counter) em que o frame lido foi capturado
        self.failed = False
        self.running = True
        self.new = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.failed = True
                self.new.set()
                return
            with self.lock:
                self.frame = f
                self.ts = time.perf_counter()
            self.new.set()

    def read(self):
        if not self.new.wait(timeout=2) or self.failed:
            return False, None
        with self.lock:
            f = self.frame
            self.frame_ts = self.ts
        self.new.clear()
        return True, f

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


_color_cache = {}


def make_background(frame, mode, args, bg_image):
    h, w = frame.shape[:2]
    if mode == "blur":
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), args.blur / 2)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "image" and bg_image is not None:
        return bg_image
    # cor sólida (calculada uma vez e reaproveitada)
    key = (h, w, args.color_bgr)
    if _color_cache.get("key") != key:
        _color_cache["key"] = key
        _color_cache["img"] = np.full((h, w, 3), args.color_bgr, np.uint8)
    return _color_cache["img"]


def load_bg_image(path, w, h):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Não consegui abrir a imagem de fundo: {path}")
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)  # "cover"
    img = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1))
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w].copy()


def list_cameras(max_index=6):
    """Abre cada índice por ~1,5 s e mostra uma prévia para você identificar a webcam."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[{i}] indisponível")
            cap.release()
            continue
        frame = None
        for _ in range(15):  # descarta os primeiros frames (costumam vir pretos)
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            print(f"[{i}] abre, mas não entrega imagem")
        else:
            h, w = frame.shape[:2]
            brilho = frame.mean()
            estado = "imagem OK" if brilho > 5 else "PRETA (provável câmera virtual)"
            print(f"[{i}] {w}x{h} - {estado}")
            cv2.putText(frame, f"camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow("teste de camera", frame)
            cv2.waitKey(1500)
        cap.release()
    cv2.destroyAllWindows()
    print("\nUse o número da câmera certa assim: python camfundo.py --preview --camera N")


LOG_MODE = False


def setup_pythonw_log():
    """No pythonw não existe console: manda saída e erros para camfundo.log."""
    global LOG_MODE
    if sys.stdout is None or sys.stderr is None:
        pasta = os.path.dirname(os.path.abspath(__file__))
        log = open(os.path.join(pasta, "camfundo.log"), "w", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
        LOG_MODE = True


def set_low_priority():
    """Prioridade abaixo do normal: o programa cede CPU para jogos e outros apps."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


FACE_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/fd2da740ebc6d0fe489d86cb55133148978cd82e/models/face_detection_yunet/" + FACE_MODEL,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + FACE_MODEL,
]


def ensure_face_model():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, FACE_MODEL)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    print("Baixando modelo de rosto YuNet (só na primeira vez)...")
    for url in FACE_URLS:
        try:
            urllib.request.urlretrieve(url, path + ".part")
            if os.path.getsize(path + ".part") > 100_000:  # descarta arquivo-ponteiro do Git LFS
                os.replace(path + ".part", path)
                return path
        except Exception as e:
            print("  falhou:", e)
    if os.path.exists(path + ".part"):
        os.remove(path + ".part")
    return None


class YuNetDetector:
    """Detector de rosto (OpenCV FaceDetectorYN). Devolve (x, y, w, h) do maior rosto."""

    def __init__(self, model_path, width=320):
        self.width = width
        self.broken = False
        self.det = cv2.FaceDetectorYN.create(model_path, "", (width, width), 0.7, 0.3, 5)

    def detect(self, frame):
        if self.broken:
            return None
        try:
            h, w = frame.shape[:2]
            s = self.width / w
            small = cv2.resize(frame, (self.width, max(16, int(round(h * s)))), interpolation=cv2.INTER_AREA)
            self.det.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.det.detect(small)
        except cv2.error:
            self.broken = True
            return None
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        return (f[0] / s, f[1] / s, f[2] / s, f[3] / s)


def build_face_detector():
    path = ensure_face_model()
    if path is None:
        print("Não consegui baixar o modelo de rosto, então o BPM pela câmera fica desligado.\n"
              f"Baixe {FACE_MODEL} do repositório opencv/opencv_zoo e coloque na pasta models/.")
        return None
    try:
        det = YuNetDetector(path)
        det.det.setInputSize((320, 180))
        det.det.detect(np.zeros((180, 320, 3), np.uint8))  # teste de funcionamento
        return det
    except Exception as e:
        print(f"Falha ao iniciar o detector de rosto ({e}).\n"
              f"Seu OpenCV é o {cv2.__version__}. O modelo usado foi feito para o OpenCV 4.x; instale com:\n"
              '    python -m pip install "opencv-python>=4.8,<5"')
        return None


def pos_signal(rgb, fs):
    """Método POS (Wang et al., 2017): combina R, G e B para isolar o pulso e cancelar movimento.

    Versão vetorizada: todas as janelas de 1,6 s são calculadas de uma vez e somadas com sobreposição.
    """
    n, l = len(rgb), int(1.6 * fs)
    if n < l:
        return np.zeros(n)
    win = np.lib.stride_tricks.sliding_window_view(rgb, l, axis=0)          # (m, 3, l)
    cn = win / (win.mean(axis=2, keepdims=True) + 1e-9)
    s0 = cn[:, 1] - cn[:, 2]                                                # G - B
    s1 = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]                              # -2R + G + B
    h = s0 + (s0.std(axis=1, keepdims=True) / (s1.std(axis=1, keepdims=True) + 1e-9)) * s1
    h -= h.mean(axis=1, keepdims=True)
    m = h.shape[0]
    out = np.zeros(n)
    for j in range(l):                                                      # soma com sobreposição
        out[j:j + m] += h[:, j]
    # limita picos isolados (movimento, piscadas de luz) para não dominarem o espectro
    k = 3.5 * 1.4826 * np.median(np.abs(out - np.median(out))) + 1e-12
    return np.clip(out, -k, k)


def power_spectrum(x, fs):
    """Espectro de potência (com zero-padding) e o eixo de frequências em bpm."""
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    p = np.abs(np.fft.rfft(x * np.hanning(n), nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1.0 / fs) * 60.0, p


def peak_from_spectrum(f, p, lo_bpm, hi_bpm, around=None, radius=15.0):
    """Maior pico (máximo local de verdade) entre lo_bpm e hi_bpm; com `around`, só perto desse valor.

    Devolve (bpm, snr) ou None. Sem `around`, o ruído é medido em toda a faixa; com `around`, só
    numa vizinhança de ±40 bpm do pico, para que uma interferência distante não derrube a leitura.
    """
    band = np.nonzero((f >= lo_bpm) & (f <= hi_bpm))[0]
    if len(band) < 5:
        return None
    cand = band if around is None else band[np.abs(f[band] - around) <= radius]
    cand = cand[(cand > 0) & (cand < len(p) - 1)]
    cand = cand[(p[cand] >= p[cand - 1]) & (p[cand] >= p[cand + 1])]   # exclui "ladeiras" de outro pico
    if len(cand) == 0:
        return None
    k = cand[np.argmax(p[cand])]
    a, b, c = np.log(p[k - 1] + 1e-12), np.log(p[k] + 1e-12), np.log(p[k + 1] + 1e-12)   # refinamento parabólico
    d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    peak = f[k] + float(np.clip(d, -1, 1)) * (f[1] - f[0])
    dist = np.abs(f[band] - peak)
    near = dist <= 10.0
    scope = np.ones_like(near) if around is None else dist <= 40.0
    snr = p[band][near].sum() / max(p[band][~near & scope].sum(), 1e-12)
    return float(peak), float(snr)


def spectral_peak(x, fs, lo_bpm, hi_bpm, around=None, radius=15.0):
    f, p = power_spectrum(x, fs)
    return peak_from_spectrum(f, p, lo_bpm, hi_bpm, around, radius)


class HeartRate:
    """Estima o BPM pela variação de cor da pele do rosto (rPPG: testa e bochechas + método POS).

    Melhorias de estabilidade: cada região (testa, bochecha esquerda, bochecha direita) é analisada
    separadamente e as melhores pesam mais; depois de uma leitura confiável o programa "trava" na
    frequência e só acompanha mudanças graduais (±15 bpm), em vez de saltar para ruído de outra faixa.
    """

    REGIONS = [(0.30, 0.05, 0.70, 0.27), (0.12, 0.50, 0.38, 0.74), (0.62, 0.50, 0.88, 0.74)]

    def __init__(self, detector, window=15.0, min_seconds=10.0, lo=45, hi=180,
                 snr_min=0.8, detect_every=2, hold=10.0):
        self.detector, self.window, self.min_seconds = detector, window, min_seconds
        self.lo, self.hi, self.snr_min = lo, hi, snr_min
        self.detect_every, self.hold = detect_every, hold
        self.buf = deque()
        self.box = None
        self.rois = []
        self.n = 0
        self.miss_since = None
        self.last_est = None
        self.bpm, self.bpm_ts, self.cands, self.alt = None, -1e9, [], []
        self.last_good, self.last_good_ts = None, -1e9   # última leitura confiável (referência de plausibilidade)
        self.status = "medindo..."
        self.last_snr, self.last_raw = 0.0, None

    # --- entrada de frames ---
    def update(self, frame, ts):
        self.n += 1
        if self.box is None or self.n % self.detect_every == 0:
            box = self.detector.detect(frame)
            if box is None:
                if self.miss_since is None:
                    self.miss_since = ts
                if ts - self.miss_since > 1.0:
                    self.box, self.rois = None, []
                    self.buf.clear()
                    self.status = "erro no rosto" if getattr(self.detector, "broken", False) else "sem rosto"
                    return
            else:
                self.miss_since = None
                b = np.array(box, float)
                if self.box is None or np.abs(b[:2] - self.box[:2]).max() > 0.5 * self.box[2]:
                    self.box = b
                else:
                    self.box = 0.6 * self.box + 0.4 * b
        if self.box is None:
            return
        sample = self._sample(frame)
        if sample is not None:
            self.buf.append((ts,) + sample)
            while self.buf and self.buf[0][0] < ts - (self.window + 3.0):
                self.buf.popleft()
        if self.last_est is None:
            self.last_est = ts
        if ts - self.last_est >= 1.0:
            self._estimate(ts)

    def _sample(self, frame):
        """Cor média (R,G,B) de cada região. Devolve 9 valores ou None se alguma região for inválida."""
        h, w = frame.shape[:2]
        x, y, bw, bh = self.box
        vals, rois = [], []
        for fx0, fy0, fx1, fy1 in self.REGIONS:
            x0, x1 = int(max(0, x + fx0 * bw)), int(min(w, x + fx1 * bw))
            y0, y1 = int(max(0, y + fy0 * bh)), int(min(h, y + fy1 * bh))
            if x1 - x0 < 4 or y1 - y0 < 4:
                self.rois = rois
                return None
            b, g, r = cv2.mean(frame[y0:y1, x0:x1])[:3]
            rois.append((x0, y0, x1, y1))
            if not 20 < (b + g + r) / 3 < 245:       # região escura demais ou estourada
                self.rois = rois
                return None
            vals += [r, g, b]
        self.rois = rois
        return tuple(vals)

    # --- estimativa ---
    def _estimate(self, now):
        self.last_est = now
        arr = np.array(self.buf)
        if len(arr) < 30 or arr[-1, 0] - arr[0, 0] < self.min_seconds:
            self.status = "medindo..."
            return
        t = arr[:, 0]
        fs = 30.0
        tu = np.arange(max(t[0], t[-1] - self.window), t[-1], 1.0 / fs)
        spectra, weights = [], []
        for r in range(3):
            rgb = np.stack([np.interp(tu, t, arr[:, 1 + 3 * r + k]) for k in range(3)], axis=1)
            f, p = power_spectrum(pos_signal(rgb, fs), fs)
            pb = p[(f >= self.lo) & (f <= self.hi)]
            weights.append(pb.max() / (pb.mean() + 1e-12))   # "nitidez" do pico: quanto melhor a região, mais pesa
            spectra.append(p / (pb.sum() + 1e-12))
        w = np.array(weights) / np.sum(weights)
        spec = sum(wi * si for wi, si in zip(w, spectra))

        full = peak_from_spectrum(f, spec, self.lo, self.hi)
        if full is None:
            return
        self.last_raw, self.last_snr = full
        if self.value(now) is not None:
            self._track(f, spec, full, now)
            return
        # sem leitura travada: busca. Se houve leitura confiável há menos de 60 s, só aceita picos
        # plausíveis (±25 bpm dela); na dúvida mostra "sinal fraco" em vez de um valor absurdo.
        limite = self.snr_min
        if self.last_good is not None and now - self.last_good_ts < 60.0:
            res = peak_from_spectrum(f, spec, self.lo, self.hi, around=self.last_good, radius=25.0)
            limite = self.snr_min * 1.25
        else:
            res = full
        if res is None or res[1] < limite or res[0] < self.lo + 5 or res[0] > self.hi - 5:   # colado na borda = ruído
            self.cands, self.status = [], "sinal fraco"
            return
        self.status = "ok"
        self.cands = (self.cands + [res[0]])[-3:]
        if len(self.cands) == 3 and max(self.cands) - min(self.cands) <= 6:
            self.bpm, self.bpm_ts, self.cands, self.alt = float(np.median(self.cands)), now, [], []
            self.last_good, self.last_good_ts = self.bpm, now

    def _track(self, f, spec, full, now):
        """Com leitura travada: só acompanha o pico perto do valor atual (±15 bpm)."""
        tr = peak_from_spectrum(f, spec, self.lo, self.hi, around=self.bpm, radius=15.0)
        if tr is not None and tr[1] >= self.snr_min:
            self.status = "ok"
            self.last_raw, self.last_snr = tr
            self.bpm += 0.3 * (tr[0] - self.bpm)
            self.bpm_ts = now
            self.last_good, self.last_good_ts = self.bpm, now
        else:
            self.status = "sinal fraco"
        # Troca de frequência (caso a trava tenha sido em ruído): só se OUTRO pico bem forte persistir
        # por ~30 estimativas seguidas. Mudanças reais de pulso são graduais e o acompanhamento já cobre.
        peak, snr = full
        if abs(peak - self.bpm) > 15 and snr >= self.snr_min * 2 and self.lo + 5 <= peak <= self.hi - 5:
            self.alt = (self.alt + [peak])[-30:]
            if len(self.alt) == 30 and max(self.alt) - min(self.alt) <= 8:
                self.bpm, self.bpm_ts, self.alt = float(np.median(self.alt)), now, []
                self.last_good, self.last_good_ts = self.bpm, now
        else:
            self.alt = []

    # --- saída ---
    def value(self, now):
        if self.bpm is not None and now - self.bpm_ts <= self.hold:
            return self.bpm
        return None

    def message(self, now):
        return "--" if self.value(now) is not None else self.status

    def draw_debug(self, img):
        if self.box is not None:
            x, y, w, h = (int(v) for v in self.box)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for x0, y0, x1, y1 in self.rois:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)
        raw = "-" if self.last_raw is None else f"{self.last_raw:.0f}"
        trava = "travado" if self.bpm is not None and self.bpm_ts > -1e8 else "buscando"
        cv2.putText(img, f"rPPG: bruto {raw} | snr {self.last_snr:.2f} | {self.status} | {trava}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


class HeartLabel:
    """Label inferior central: coração que bate no ritmo do BPM (gelado = baixo, quente = alto)."""

    # (posição 0..1 na faixa bpm_min..bpm_max, cor RGB): gelo -> rosa -> vermelho
    STOPS = [(0.0, (120, 200, 255)), (0.5, (255, 105, 150)), (1.0, (255, 60, 30))]

    def __init__(self, bpm_min=50, bpm_max=120, show_number=True):
        self.bpm_min = float(bpm_min)
        self.bpm_max = max(float(bpm_max), self.bpm_min + 1.0)
        self.show_number = show_number
        self.phase = 0.0       # fase da batida (contínua: o ritmo muda sem "pular")
        self.shown = None      # BPM suavizado
        self.last_t = None
        t = np.linspace(0, 2 * np.pi, 90)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.stack([x, y], axis=1)
        pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
        self.unit = pts / np.abs(pts).max()

    def color(self, bpm):
        u = float(np.clip((bpm - self.bpm_min) / (self.bpm_max - self.bpm_min), 0.0, 1.0))
        for (u0, c0), (u1, c1) in zip(self.STOPS, self.STOPS[1:]):
            if u <= u1:
                f = (u - u0) / (u1 - u0)
                r, g, b = (c0[i] + (c1[i] - c0[i]) * f for i in range(3))
                break
        return (int(b), int(g), int(r))  # BGR

    @staticmethod
    def beat(seconds):
        """Escala do coração: batida dupla ("tum-tum") ao longo de um ciclo."""
        p1 = np.exp(-((seconds - 0.08) / 0.07) ** 2)
        p2 = np.exp(-((seconds - 0.30) / 0.07) ** 2)
        return 1.0 + 0.30 * p1 + 0.18 * p2

    def draw(self, img, bpm, now, msg="--"):
        h, w = img.shape[:2]
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 0.25)
        self.last_t = now

        if bpm is None:                      # sem leitura: coração cinza e parado
            color, scale, label = (150, 150, 150), 1.0, msg
        else:
            bpm = float(np.clip(bpm, 30, 200))
            self.shown = bpm if self.shown is None else self.shown + (bpm - self.shown) * min(1.0, dt * 3.0)
            self.phase = (self.phase + dt * self.shown / 60.0) % 1.0
            scale = self.beat(self.phase * 60.0 / self.shown)
            color, label = self.color(self.shown), f"{int(round(self.shown))} BPM"
        if not self.show_number:
            label = ""

        base = h * 0.034                     # meia-largura do coração em repouso
        reserve = base * 1.32                # espaço para o pico da batida
        font = cv2.FONT_HERSHEY_DUPLEX
        fs, th = h / 720 * 0.85, max(1, int(round(h / 720 * 2)))
        tw, tht = (cv2.getTextSize(label, font, fs, th)[0] if label else (0, 0))
        gap = base * 0.7 if label else 0.0
        pad = base * 0.75
        pill_w = int(2 * reserve + gap + tw + 2 * pad)
        pill_h = int(2 * reserve + 1.2 * pad)
        x0 = max(0, (w - pill_w) // 2)
        y1 = h - int(h * 0.035)
        y0 = max(0, y1 - pill_h)
        x1 = min(w, x0 + pill_w)

        # fundo escuro translúcido em formato de pílula (só na região pequena do label)
        roi = img[y0:y1, x0:x1]
        mh, mw = roi.shape[:2]
        mask = np.zeros((mh, mw), np.uint8)
        r = mh // 2
        cv2.rectangle(mask, (r, 0), (max(r, mw - r - 1), mh - 1), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (max(r, mw - r - 1), r), r, 255, -1, cv2.LINE_AA)
        a = (cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0 * 0.5)[..., None]
        roi[:] = (roi * (1.0 - a)).astype(np.uint8)

        cx, cy = x0 + pad + reserve, (y0 + y1) / 2.0
        pts = (self.unit * (base * scale) + np.array([cx, cy])).astype(np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, tuple(int(c * 0.6) for c in color), 1, cv2.LINE_AA)
        if label:
            org = (int(x0 + pad + 2 * reserve + gap), int(cy + tht / 2))
            cv2.putText(img, label, org, font, fs, (255, 255, 255), th, cv2.LINE_AA)


def parse_color(hexstr):
    hexstr = hexstr.lstrip("#")
    r, g, b = (int(hexstr[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def main():
    p = argparse.ArgumentParser(description="Remoção de fundo de webcam com IA")
    p.add_argument("--camera", type=int, default=0, help="índice da webcam")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", choices=MODEL_FILES, default="mobilenet")
    p.add_argument("--device", choices=["dml", "cpu"], default="dml",
                   help="dml = GPU AMD/Intel/Nvidia via DirectML")
    p.add_argument("--infer-width", type=int, default=640,
                   help="largura usada pela IA (maior = bordas melhores, menos FPS)")
    p.add_argument("--ratio", type=float, default=None,
                   help="downsample_ratio do RVM (padrão: automático)")
    p.add_argument("--mode", choices=["blur", "image", "color", "none"], default="color")
    p.add_argument("--image", help="imagem de fundo para --mode image")
    p.add_argument("--color", default="00B140", help="cor hex para --mode color")
    p.add_argument("--blur", type=float, default=30, help="intensidade do desfoque")
    p.add_argument("--edge", type=float, default=0.08,
                   help="limpeza das bordas da máscara (0 a 0.3)")
    p.add_argument("--shrink", type=int, default=0,
                   help="encolhe a máscara N pixels (tira halo em volta)")
    p.add_argument("--mirror", action="store_true", help="espelha a imagem")
    p.add_argument("--preview", action="store_true", help="mostra janela de preview")
    p.add_argument("--no-vcam", action="store_true", help="não envia à câmera virtual")
    p.add_argument("--paralelo", action="store_true",
                   help="roda a IA em paralelo: saída fluida, máscara pode atrasar ~1 frame")
    p.add_argument("--ia-fps", type=float, default=0,
                   help="limita quantas vezes por segundo a IA roda no modo --paralelo (0 = sem limite)")
    p.add_argument("--fluxo", action="store_true",
                   help="compensa o movimento da máscara velha (útil com --ia-fps baixo ou --skip)")
    p.add_argument("--threads", type=int, default=0,
                   help="limita as threads de CPU usadas (OpenCV e ONNX); 0 = automático")
    p.add_argument("--prioridade", choices=["normal", "baixa"], default="normal",
                   help="baixa = o programa cede CPU para jogos e outros apps")
    p.add_argument("--skip", type=int, default=1,
                   help="roda a IA a cada N frames e reaproveita a máscara (2 = ~2x mais FPS)")
    p.add_argument("--profile", action="store_true",
                   help="mostra o tempo (ms) de cada etapa por frame")
    p.add_argument("--coracao", action="store_true",
                   help="mostra embaixo um coração que bate no ritmo do BPM")
    p.add_argument("--bpm-teste", type=float, default=None,
                   help="BPM manual para o coração (padrão 75; ativa --coracao)")
    p.add_argument("--bpm-min", type=float, default=50, help="BPM da cor mais gelada")
    p.add_argument("--bpm-max", type=float, default=120, help="BPM da cor mais quente")
    p.add_argument("--bpm-sem-numero", action="store_true", help="coração sem o número")
    p.add_argument("--bpm-camera", action="store_true",
                   help="mede o BPM pela câmera (rPPG) e mostra no coração")
    p.add_argument("--bpm-snr", type=float, default=0.8,
                   help="confiança mínima para aceitar a leitura (menor = mais sensível, mais erros)")
    p.add_argument("--bpm-debug", action="store_true",
                   help="mostra no preview o rosto, as regiões medidas e a confiança")
    p.add_argument("--list-cameras", action="store_true",
                   help="testa os índices de câmera e mostra uma prévia de cada")
    setup_pythonw_log()
    args = p.parse_args()
    if args.list_cameras:
        list_cameras()
        return
    args.color_bgr = parse_color(args.color)
    if args.threads > 0:
        cv2.setNumThreads(args.threads)
    if args.prioridade == "baixa":
        set_low_priority()
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camfundo.pid")

    matter = Matter(ensure_model(args.model), args.infer_width, args.ratio,
                    args.device, args.edge, args.shrink, args.threads)

    hr = None
    if args.bpm_camera:
        det = build_face_detector()
        hr = HeartRate(det, snr_min=args.bpm_snr) if det is not None else None

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        sys.exit("Não consegui abrir a webcam. Tente outro --camera (1, 2...).")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {w}x{h}")

    bg_image = load_bg_image(args.image, w, h) if args.image else None
    mode = args.mode
    if mode == "image" and bg_image is None:
        sys.exit("--mode image precisa de --image caminho.jpg")

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=w, height=h, fps=args.fps,
                                       fmt=pyvirtualcam.PixelFormat.BGR)
            print("Câmera virtual:", vcam.device)
        except Exception as e:
            sys.exit(f"Falha ao abrir a câmera virtual: {e}\n"
                     "Instale o OBS Studio (ele inclui a Virtual Camera) e tente de novo.")
    if vcam is None and not args.preview:
        args.preview = True

    with open(pid_file, "w") as f:  # só existe enquanto roda (usado pelo parar_camfundo.bat)
        f.write(str(os.getpid()))
    grab = Grabber(cap)
    a8, n = None, 0
    ref, fid = None, -1
    amatter = AsyncMatter(matter, args.ia_fps) if args.paralelo else None
    last_ai = 0
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    next_t = time.perf_counter()
    heart = (HeartLabel(args.bpm_min, args.bpm_max, not args.bpm_sem_numero)
             if (args.coracao or args.bpm_teste is not None or args.bpm_camera) else None)
    test_bpm = args.bpm_teste if args.bpm_teste is not None else 75.0
    show_heart = True
    pc = time.perf_counter
    acc = {"leitura": 0.0, "ia": 0.0, "fluxo": 0.0, "composicao": 0.0, "overlay": 0.0, "bpm": 0.0, "envio": 0.0}
    t0, frames, fps_txt = time.time(), 0, ""
    try:
        while True:
            t = pc()
            ok, frame = grab.read()
            if not ok:
                print("Falha ao ler frame da webcam.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            acc["leitura"] += pc() - t

            if hr is not None:
                t = pc()
                hr.update(frame, grab.frame_ts)
                acc["bpm"] += pc() - t

            if mode == "none":
                out = frame
            else:
                t = pc()
                small = small_gray(frame) if args.fluxo else None
                if amatter is not None:
                    amatter.submit(frame, small, n)
                    a8, ref, fid = amatter.latest()
                elif a8 is None or n % args.skip == 0:
                    a8 = matter.alpha(frame)
                    ref, fid = small, n
                acc["ia"] += pc() - t
                if args.fluxo and fid != n and ref is not None:
                    t = pc()
                    a8 = warp_mask(a8, ref, small)
                    acc["fluxo"] += pc() - t
                n += 1
                t = pc()
                bg = make_background(frame, mode, args, bg_image)
                out = compose(frame, a8, bg)
                acc["composicao"] += pc() - t

            if heart is not None and show_heart:
                t = pc()
                if args.bpm_camera:
                    heart.draw(out, hr.value(t) if hr else None, t,
                               hr.message(t) if hr else "sem modelo")
                else:
                    heart.draw(out, test_bpm, t)
                acc["overlay"] += pc() - t

            t = pc()
            if vcam is not None:
                vcam.send(out)
            acc["envio"] += pc() - t

            frames += 1
            if time.time() - t0 >= 1.0:
                fps_txt = f"{frames / (time.time() - t0):.1f} FPS"
                if args.profile:
                    ms = " | ".join(f"{k} {v / frames * 1000:.1f} ms" for k, v in acc.items())
                    if amatter is not None:
                        ms += f" | IA {amatter.count - last_ai}/s"
                        last_ai = amatter.count
                    print(f"{fps_txt} | {ms}")
                elif not LOG_MODE:
                    print(fps_txt, end="\r")
                acc = {k: 0.0 for k in acc}
                t0, frames = time.time(), 0

            if args.preview:
                view = out.copy()
                if hr is not None and args.bpm_debug:
                    hr.draw_debug(view)
                cv2.putText(view, f"{mode} | {fps_txt}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("camfundo", view)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("b"):
                    mode = "blur"
                elif k == ord("c"):
                    mode = "color"
                elif k == ord("i") and bg_image is not None:
                    mode = "image"
                elif k == ord("n"):
                    mode = "none"
                elif k in (ord("+"), ord("=")):
                    test_bpm = min(200.0, test_bpm + 5)
                elif k in (ord("-"), ord("_")):
                    test_bpm = max(30.0, test_bpm - 5)
                elif k == ord("h"):
                    show_heart = not show_heart

            # limita a taxa de processamento ao valor de --fps
            if period > 0:
                next_t += period
                delay = next_t - pc()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = pc()
    except KeyboardInterrupt:
        pass
    finally:
        grab.stop()
        try:
            os.remove(pid_file)
        except OSError:
            pass
        if amatter is not None:
            amatter.stop()
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
