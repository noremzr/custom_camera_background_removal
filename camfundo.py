#!/usr/bin/env python3
"""
camfundo - remoção de fundo da webcam com IA (alternativa ao Nvidia Broadcast).

Pipeline: webcam -> Robust Video Matting (ONNX) na GPU via DirectML
          -> composição (blur / imagem / cor) -> câmera virtual do OBS.

Uso rápido:
    python camfundo.py --preview
    python camfundo.py --mode image --image fundo.jpg
Teclas na janela de preview: b = blur, c = cor, i = imagem, n = sem efeito, q/Esc = sair
"""
import argparse
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
            self.new.set()

    def read(self):
        if not self.new.wait(timeout=2) or self.failed:
            return False, None
        with self.lock:
            f = self.frame
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
    pc = time.perf_counter
    acc = {"leitura": 0.0, "ia": 0.0, "fluxo": 0.0, "composicao": 0.0, "envio": 0.0}
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
