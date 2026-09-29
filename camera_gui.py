"""
Interface de capture photo pour cameras USB haute resolution.

Les resolutions proposees ne sont que des propositions : chaque camera a sa
propre liste de modes, et "Detecter les resolutions" remplace les boutons par
les modes que la camera selectionnee fournit reellement. Demander un mode
inexistant ne provoque pas d'erreur cote camera : elle renvoie silencieusement
un format plus petit, d'ou l'importance de la detection.

Architecture : tout l'acces camera (ouverture, lecture, capture, detection)
se fait dans un thread dedie. Le thread Tk ne fait qu'afficher la derniere
image disponible et consommer des evenements. L'interface ne se fige donc
jamais, meme pendant une lecture de frame 48MP (~0,3 s) ou une reouverture
du peripherique (~1 s).

Dependances : opencv-python, pillow, numpy
Lancement   : python camera_gui.py
"""

import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# Resolutions par defaut (a ajuster si la camera utilise d'autres valeurs
# exactes : utiliser le bouton "Detecter les resolutions" pour verifier).
# Prereglages affiches au demarrage. Ce ne sont que des propositions : chaque
# camera a sa propre liste de modes et "Detecter les resolutions" remplace
# ceux-ci par les modes reellement fournis par la camera selectionnee.
DEFAULT_PRESETS = [
    (1920, 1080), (3840, 2160), (4000, 3000), (5472, 3648), (8000, 6000),
]

# Taille par defaut de la zone d'apercu. L'image est redimensionnee en
# conservant ses proportions (pas de deformation).
PREVIEW_DISPLAY_SIZE = (640, 480)
PREVIEW_MIN_SIZE = (320, 240)
PREVIEW_MAX_SIZE = (1280, 960)

# Resolutions candidates testees par le bouton "Detecter". La liste couvre les
# modes courants des cameras UVC, y compris les petits : une camera peut fort
# bien n'offrir que 1280x720 et 5472x3648, sans rien entre les deux.
CANDIDATE_RESOLUTIONS = [
    (1280, 720), (1920, 1080), (2048, 1536), (2560, 1440), (2592, 1944),
    (3264, 2448), (3840, 2160), (4000, 3000), (4032, 3024), (4656, 3496),
    (5120, 3840), (5184, 3888), (5472, 3648), (6000, 4000), (6400, 4800),
    (8000, 6000), (8064, 6048),
]

MAX_CAMERA_INDEX = 5                # indices sondes par le scan
MAX_CONSECUTIVE_READ_ERRORS = 40
JPEG_QUALITY = 95
BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

# Formats tentes a l'ouverture, dans l'ordre. None = format natif annonce par
# la camera. Voir _open_device : imposer MJPG a l'aveugle fait retomber les
# capteurs non compresses sur un petit mode.
FOURCC_ATTEMPTS = [None, "MJPG"]

# Duree laissee a la camera pour basculer dans le mode demande apres
# l'ouverture, et nombre d'images de meme taille a partir duquel on considere
# que la bascule est terminee. Voir CameraWorker._settle : une camera 20MP
# commence par renvoyer quelques images en 1280x720 avant de passer en pleine
# resolution, et elle n'y arrive qu'a une poignee d'images par seconde.
OPEN_SETTLE_SECONDS = 5.0
OPEN_STABLE_FRAMES = 3


# --------------------------------------------------------------------------
# Traitement image
# --------------------------------------------------------------------------

def fast_thumbnail_rgb(frame_bgr, target_size):
    """Reduit une frame BGR vers un apercu RGB, en conservant les proportions.

    Un redimensionnement direct d'une frame 8000x6000 (LANCZOS de Pillow dans
    la version precedente) coutait plusieurs centaines de millisecondes par
    image et rendait l'apercu inutilisable. On decime d'abord par pas entier
    (simple parcours memoire), puis on finit en INTER_AREA sur une image deja
    petite : meme rendu visuel pour une fraction du cout.
    """
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    tw, th = target_size
    scale = min(tw / float(w), th / float(h))

    if scale < 0.5:
        # on garde un facteur >= 2 pour l'etape INTER_AREA qui suit, afin de
        # limiter l'aliasing introduit par la decimation.
        step = max(1, int(1.0 / scale) // 2)
        if step > 1:
            frame_bgr = np.ascontiguousarray(frame_bgr[::step, ::step])
            h, w = frame_bgr.shape[:2]
            scale = min(tw / float(w), th / float(h))

    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(frame_bgr, (out_w, out_h), interpolation=interp)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def fmt_res(res):
    """Cle texte d'une resolution, utilisee comme valeur des boutons radio."""
    return "{}x{}".format(res[0], res[1])


def parse_res(text):
    """Inverse de fmt_res. Renvoie None si le texte n'est pas une resolution."""
    try:
        width, height = text.lower().split("x", 1)
        return int(width), int(height)
    except (AttributeError, ValueError):
        return None


def unique_path(directory, width, height):
    """Nom de fichier horodate a la milliseconde, garanti non existant.

    L'horodatage a la seconde de la version precedente ecrasait
    silencieusement une photo quand deux captures tombaient dans la meme
    seconde.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = "photo_{}x{}_{}".format(width, height, stamp)
    path = os.path.join(directory, base + ".jpg")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, "{}_{}.jpg".format(base, counter))
        counter += 1
    return path


def imwrite_unicode(path, frame, quality=JPEG_QUALITY):
    """Ecrit un JPEG en supportant les chemins non-ASCII.

    cv2.imwrite echoue silencieusement (il renvoie False) quand le chemin
    contient des accents sous Windows : on encode en memoire, puis on ecrit
    le fichier nous-memes.
    """
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError("Encodage JPEG impossible.")
    with open(path, "wb") as handle:
        handle.write(buf.tobytes())


# --------------------------------------------------------------------------
# Thread camera
# --------------------------------------------------------------------------

class CameraWorker(threading.Thread):
    """Proprietaire exclusif de l'objet cv2.VideoCapture.

    Les commandes arrivent par une file, les reponses repartent par une autre
    file consommee par le thread Tk. Aucun appel Tk n'est fait ici.
    """

    def __init__(self, events):
        super().__init__(daemon=True)
        self._cmds = queue.Queue()
        self._events = events
        self._stop_flag = threading.Event()

        self._cap = None
        self._index = None
        self._res = (0, 0)
        self._streaming = False
        self._read_errors = 0

        self._preview_size = PREVIEW_DISPLAY_SIZE
        self._frame_lock = threading.Lock()
        self._latest = None             # apercu RGB pret a afficher
        self._latest_shape = None       # resolution reelle du capteur
        self._fps = 0.0
        self._last_frame_time = None

    # -- API appelee depuis le thread Tk -----------------------------------

    def send(self, name, **kwargs):
        self._cmds.put((name, kwargs))

    def set_preview_size(self, size):
        self._preview_size = size

    def take_preview(self):
        """Renvoie le dernier apercu disponible (None s'il a deja ete affiche)."""
        with self._frame_lock:
            frame, shape, fps = self._latest, self._latest_shape, self._fps
            self._latest = None
        return frame, shape, fps

    def shutdown(self):
        self._stop_flag.set()
        self._cmds.put(("__stop__", {}))

    # -- Boucle du thread --------------------------------------------------

    def run(self):
        while not self._stop_flag.is_set():
            try:
                if self._streaming and self._cap is not None:
                    item = self._cmds.get_nowait()
                else:
                    item = self._cmds.get(timeout=0.15)
            except queue.Empty:
                item = None

            if item is not None:
                name, kwargs = item
                if name == "__stop__":
                    break
                try:
                    getattr(self, "_do_" + name)(**kwargs)
                except Exception as exc:                       # noqa: BLE001
                    self._emit("error", message="{} : {}".format(name, exc))
                    self._emit("busy", value=False)
                continue

            if self._streaming and self._cap is not None:
                self._read_preview()

        self._release()

    # -- Utilitaires internes ----------------------------------------------

    def _emit(self, kind, **payload):
        self._events.put((kind, payload))

    def _emit_state(self):
        """Publie l'etat reel : un peripherique peut etre ouvert sans que
        l'apercu tourne (capture faite alors que l'apercu etait arrete). Le
        bouton "Fermer la camera" doit rester actif dans ce cas."""
        self._emit("state", streaming=self._streaming, device=self._cap is not None)

    def _release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:                                  # noqa: BLE001
                pass
        self._cap = None
        self._res = (0, 0)
        # On jette l'image en attente : sans cela, une frame publiee juste
        # avant la fermeture etait encore affichee apres, a la place de
        # l'ecran d'attente.
        with self._frame_lock:
            self._latest = None
            self._latest_shape = None
        self._fps = 0.0
        self._last_frame_time = None

    def _open_device(self, index, width, height, announce=True):
        """Ouvre la camera a la resolution voulue, en negociant le format.

        Changer la resolution sur un flux deja actif fige certaines cameras
        UVC bon marche : on ferme et on rouvre systematiquement.

        Le format ne peut pas etre impose a l'aveugle. Les capteurs USB3 non
        compresses ne donnent leur pleine resolution qu'en format natif,
        tandis que les modules 48MP bon marche n'exposent leurs grands modes
        qu'en MJPG. On essaie donc les deux et on garde celui qui rend
        exactement la resolution demandee.
        """
        self._release()
        best_cap = None
        best_frame = None
        for backend in BACKENDS:
            for fourcc in FOURCC_ATTEMPTS:
                cap, frame = self._try_open(index, backend, width, height, fourcc)
                if cap is None:
                    continue
                if (frame.shape[1], frame.shape[0]) == (width, height):
                    if best_cap is not None:
                        best_cap.release()
                    self._install(cap, index, frame)
                    return True
                if best_frame is None or frame.size > best_frame.size:
                    if best_cap is not None:
                        best_cap.release()
                    best_cap, best_frame = cap, frame
                else:
                    cap.release()
            if best_cap is not None:
                # Ce backend repond : inutile d'essayer les suivants, qui
                # ajouteraient une seconde d'attente pour le meme resultat.
                break

        if best_cap is not None:
            self._install(best_cap, index, best_frame)
            return True
        if announce:
            self._emit("error", message="Impossible d'ouvrir la camera {}.".format(index))
        return False

    def _try_open(self, index, backend, width, height, fourcc):
        """Une tentative d'ouverture. Renvoie (capture, image stabilisee)."""
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception:                                      # noqa: BLE001
            return None, None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None, None
        try:
            if fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # evite un apercu en retard
        except Exception:                                      # noqa: BLE001
            pass

        frame = self._settle(cap, width, height)
        if frame is None:
            cap.release()
            return None, None
        return cap, frame

    @staticmethod
    def _settle(cap, width, height):
        """Attend que la camera ait reellement bascule dans le mode demande.

        Le changement de mode est differe : les premieres images qui suivent
        l'ouverture arrivent encore a l'ancienne resolution, typiquement
        1280x720. Juger sur la premiere image fait donc conclure a tort que la
        camera refuse la resolution demandee, et fige l'application sur ce
        petit format alors que le flux passe en pleine resolution juste apres.

        On lit donc jusqu'a obtenir la taille demandee, ou jusqu'a ce que la
        taille se repete assez pour etre consideree comme definitive.
        """
        frame = None
        stable = 0
        deadline = time.perf_counter() + OPEN_SETTLE_SECONDS
        while time.perf_counter() < deadline:
            try:
                ok, candidate = cap.read()
            except Exception:                                  # noqa: BLE001
                ok, candidate = False, None
            if not ok or candidate is None or candidate.size == 0:
                time.sleep(0.02)
                continue
            if (candidate.shape[1], candidate.shape[0]) == (width, height):
                return candidate
            if frame is not None and candidate.shape == frame.shape:
                stable += 1
                if stable >= OPEN_STABLE_FRAMES:
                    return candidate
            else:
                stable = 0
            frame = candidate
        return frame

    def _install(self, cap, index, frame):
        self._cap = cap
        self._index = index
        self._res = (frame.shape[1], frame.shape[0])
        self._read_errors = 0
        self._last_frame_time = None
        self._publish(frame)

    @staticmethod
    def _resolution_note(got, want):
        """Explique un ecart entre resolution demandee et resolution obtenue."""
        if got == want:
            return ""
        note = " La camera a impose {}x{} au lieu de {}x{}.".format(
            got[0], got[1], want[0], want[1])
        if got[0] * got[1] * 4 < want[0] * want[1]:
            note += (" Un ecart aussi important vient presque toujours du lien"
                     " USB : une camera USB 3 branchee sur un port ou avec un"
                     " cable USB 2 n'expose plus ses modes haute resolution.")
        return note

    def _publish(self, frame):
        thumb = fast_thumbnail_rgb(frame, self._preview_size)
        if thumb is None:
            return
        now = time.perf_counter()
        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                instant = 1.0 / delta
                self._fps = instant if self._fps == 0 else (0.8 * self._fps + 0.2 * instant)
        self._last_frame_time = now
        with self._frame_lock:
            self._latest = thumb
            self._latest_shape = (frame.shape[1], frame.shape[0])

    def _read_preview(self):
        try:
            ok, frame = self._cap.read()
        except Exception as exc:                               # noqa: BLE001
            ok, frame = False, None
            self._emit("error", message="Lecture camera : {}".format(exc))

        if not ok or frame is None or frame.size == 0:
            self._read_errors += 1
            if self._read_errors >= MAX_CONSECUTIVE_READ_ERRORS:
                self._streaming = False
                self._release()
                self._emit_state()
                self._emit("error", message="Flux camera perdu. Rebranche la camera "
                                            "puis clique sur Ouvrir / Apercu.")
            time.sleep(0.02)
            return

        self._read_errors = 0
        self._publish(frame)

    def _grab_stable_frame(self, attempts, settle):
        """Vide le buffer et renvoie la derniere frame valide.

        La version precedente ecrasait `ok` a chaque tour de boucle : un
        dernier read rate faisait echouer toute la capture alors qu'une frame
        valide avait deja ete recue.
        """
        best = None
        for _ in range(attempts):
            try:
                ok, frame = self._cap.read()
            except Exception:                                  # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None and frame.size > 0:
                best = frame
            if settle:
                time.sleep(settle)
        return best

    # -- Commandes ---------------------------------------------------------

    def _do_scan(self):
        self._emit("busy", value=True)
        self._emit("status", text="Recherche des cameras...")
        old_index, old_res, was_streaming = self._index, self._res, self._streaming
        # On libere le peripherique courant avant de sonder : sans cela, le
        # scan se heurtait a la camera deja ouverte par l'apercu.
        self._streaming = False
        self._release()

        indices = []
        labels = []
        for index in range(MAX_CAMERA_INDEX):
            cap = None
            usable = False
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                usable = cap.isOpened()
            except Exception:                                  # noqa: BLE001
                usable = False
            finally:
                if cap is not None:
                    cap.release()      # libere aussi les handles non ouverts
            if not usable:
                continue
            indices.append(index)
            labels.append(str(index))

        if not labels:
            indices, labels = [0], ["0"]
        self._emit("cameras", values=labels)

        if was_streaming and old_index in indices:
            if self._open_device(old_index, old_res[0], old_res[1]):
                self._streaming = True
        self._emit_state()
        self._emit("status", text="Cameras detectees : {}. Utilise \"Detecter les "
                                  "resolutions\" pour connaitre les modes de celle "
                                  "qui est selectionnee.".format(", ".join(labels)))
        self._emit("busy", value=False)

    def _do_open(self, index, width, height, stream=True):
        self._emit("busy", value=True)
        self._emit("status", text="Ouverture de la camera {} en {}x{}...".format(index, width, height))
        self._streaming = False
        if self._open_device(index, width, height):
            self._streaming = bool(stream)
            self._emit("status", text="Apercu actif en {}x{}.{}".format(
                self._res[0], self._res[1],
                self._resolution_note(self._res, (width, height))))
        self._emit_state()
        self._emit("busy", value=False)

    def _do_close(self):
        self._streaming = False
        self._release()
        self._emit_state()
        self._emit("status", text="Camera fermee.")

    def _do_capture(self, index, width, height, save_dir):
        self._emit("busy", value=True)
        started = time.perf_counter()
        was_streaming = self._streaming
        self._streaming = False

        # Cas courant : l'apercu tourne deja a la resolution de capture. Il
        # est alors inutile de fermer puis rouvrir la camera deux fois, ce que
        # l'ancienne version faisait systematiquement (environ 2 s perdues par
        # photo).
        reopened = False
        if self._cap is None or self._res != (width, height):
            self._emit("status", text="Passage en {}x{}...".format(width, height))
            if not self._open_device(index, width, height):
                self._emit_state()
                self._emit("busy", value=False)
                return
            reopened = True

        attempts, settle = (6, 0.05) if reopened else (2, 0.0)
        frame = self._grab_stable_frame(attempts, settle)

        if frame is None:
            self._emit("error", message="Echec de la capture : aucune image valide recue.")
        else:
            actual_w, actual_h = frame.shape[1], frame.shape[0]
            path = unique_path(save_dir, actual_w, actual_h)
            try:
                imwrite_unicode(path, frame)
            except Exception as exc:                           # noqa: BLE001
                self._emit("error", message="Enregistrement impossible : {}".format(exc))
            else:
                self._publish(frame)
                note = self._resolution_note((actual_w, actual_h), (width, height))
                self._emit("captured", path=path, width=actual_w, height=actual_h,
                           seconds=time.perf_counter() - started, note=note)

        # La camera est deja a la resolution demandee, qui est aussi celle de
        # l'apercu : aucune reouverture de retour n'est necessaire. L'ancienne
        # version en faisait une systematiquement (environ 1 s de plus par
        # photo) pour revenir a un mode identique.
        self._streaming = was_streaming and self._cap is not None
        self._emit_state()
        self._emit("busy", value=False)

    def _do_detect(self, index, restore_res):
        self._emit("busy", value=True)
        was_streaming = self._streaming
        self._streaming = False

        # Aucun pre-filtrage : demander une taille demesuree ne renseigne pas
        # de facon fiable sur le maximum (certains pilotes ramenent la demande
        # au plus grand mode, d'autres retombent sur leur mode par defaut).
        # Seule une resolution rendue a l'identique prouve qu'elle existe.
        candidates = list(CANDIDATE_RESOLUTIONS)
        supported = []
        total = len(candidates)
        for position, (w, h) in enumerate(candidates, start=1):
            self._emit("status", text="Test {}/{} : {}x{}...".format(position, total, w, h))
            # On ouvre reellement le peripherique et on lit une image : un
            # cap.get() juste apres un cap.set() renvoie la valeur demandee
            # meme quand la camera ne la gere pas, ce qui produisait des faux
            # positifs dans la liste des resolutions "supportees".
            if not self._open_device(index, w, h, announce=False):
                continue
            frame = self._grab_stable_frame(2, 0.03)
            if frame is None:
                continue
            got = (frame.shape[1], frame.shape[0])
            if got not in supported:
                supported.append(got)
        supported.sort(key=lambda r: r[0] * r[1])

        self._open_device(index, restore_res[0], restore_res[1])
        self._streaming = was_streaming and self._cap is not None
        self._emit_state()
        self._emit("detected", resolutions=supported)
        self._emit("busy", value=False)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capture photo 48MP")
        self.root.minsize(760, 660)

        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        os.makedirs(self.save_dir, exist_ok=True)

        self.preview_size = PREVIEW_DISPLAY_SIZE
        self.camera_open = False      # apercu en cours
        self.device_open = False      # peripherique tenu par l'application
        self.busy = False
        self._apply_job = None
        self._photo = None
        self._placeholder_shown = False

        self.events = queue.Queue()
        self.worker = CameraWorker(self.events)
        self.worker.start()

        self._build_ui()
        self._show_placeholder()

        # Une seule boucle periodique pour toute la duree de vie de
        # l'application : impossible d'en demarrer une seconde par
        # inadvertance, ce qui doublait la charge a chaque capture dans la
        # version precedente.
        self.root.after(33, self._tick)
        self.worker.send("scan")

    # -- Construction de l'interface ---------------------------------------

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(top, text="Camera :").grid(row=0, column=0, sticky="w")
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(top, textvariable=self.camera_var, width=8, state="readonly")
        self.camera_combo.grid(row=0, column=1, sticky="w", padx=4)
        self.camera_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_resolution(force=True))

        self.scan_btn = ttk.Button(top, text="Rafraichir", command=lambda: self.worker.send("scan"))
        self.scan_btn.grid(row=0, column=2, padx=4)
        self.open_btn = ttk.Button(top, text="Ouvrir / Apercu", command=self._open_preview)
        self.open_btn.grid(row=0, column=3, padx=4)
        self.close_btn = ttk.Button(top, text="Fermer la camera", command=self._close_preview)
        self.close_btn.grid(row=0, column=4, padx=4)

        # grid_propagate(False) : sans cela, une image plus grande agrandit le
        # cadre, qui agrandit la fenetre, qui agrandit l'apercu... en boucle.
        self.preview_frame = tk.Frame(self.root, background="#1e1e1e",
                                      width=PREVIEW_DISPLAY_SIZE[0], height=PREVIEW_DISPLAY_SIZE[1])
        self.preview_frame.grid(row=1, column=0, sticky="nsew", **pad)
        self.preview_frame.grid_propagate(False)
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_label = tk.Label(self.preview_frame, background="#1e1e1e", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.bind("<Configure>", self._on_preview_resize)

        res_frame = ttk.LabelFrame(self.root, text="Resolution de capture")
        res_frame.grid(row=2, column=0, sticky="ew", **pad)

        # Les prereglages sont reconstruits par "Detecter les resolutions" :
        # une liste figee ne correspond a aucune camera en particulier et fait
        # demander des modes inexistants, que la camera remplace en silence.
        self.res_var = tk.StringVar(value=fmt_res(DEFAULT_PRESETS[0]))
        self.preset_row = ttk.Frame(res_frame)
        self.preset_row.grid(row=0, column=0, columnspan=3, sticky="w")
        self._set_presets(DEFAULT_PRESETS)

        custom = ttk.Frame(res_frame)
        custom.grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        ttk.Label(custom, text="Largeur x Hauteur exacte :").grid(row=0, column=0)
        self.width_var = tk.StringVar()
        self.height_var = tk.StringVar()
        ttk.Entry(custom, textvariable=self.width_var, width=7).grid(row=0, column=1, padx=2)
        ttk.Label(custom, text="x").grid(row=0, column=2)
        ttk.Entry(custom, textvariable=self.height_var, width=7).grid(row=0, column=3, padx=2)
        self.apply_btn = ttk.Button(custom, text="Appliquer", command=lambda: self._apply_resolution(force=True))
        self.apply_btn.grid(row=0, column=4, padx=6)

        actions = ttk.Frame(self.root)
        actions.grid(row=3, column=0, sticky="ew", **pad)
        self.capture_btn = ttk.Button(actions, text="Capturer la photo", command=self.capture_photo)
        self.capture_btn.grid(row=0, column=0, padx=4)
        self.detect_btn = ttk.Button(actions, text="Detecter les resolutions", command=self.detect_resolutions)
        self.detect_btn.grid(row=0, column=1, padx=4)
        self.dir_btn = ttk.Button(actions, text="Dossier de sortie...", command=self.choose_dir)
        self.dir_btn.grid(row=0, column=2, padx=4)

        info = ttk.Frame(self.root)
        info.grid(row=4, column=0, sticky="ew", **pad)
        info.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Dossier de sauvegarde : {}".format(self.save_dir))
        ttk.Label(info, textvariable=self.status_var, wraplength=900,
                  justify="left").grid(row=0, column=0, sticky="w")
        self.fps_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.fps_var, foreground="#555").grid(row=0, column=1, sticky="e", padx=6)

        preset_w, preset_h = DEFAULT_PRESETS[0]
        self.width_var.set(str(preset_w))
        self.height_var.set(str(preset_h))
        self._update_buttons()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- Etat de l'interface -----------------------------------------------

    def _update_buttons(self):
        state = "disabled" if self.busy else "normal"
        for widget in (self.scan_btn, self.open_btn, self.apply_btn,
                       self.capture_btn, self.detect_btn, self.dir_btn):
            widget.configure(state=state)
        self.close_btn.configure(state="normal" if (not self.busy and self.device_open) else "disabled")
        self.camera_combo.configure(state="disabled" if self.busy else "readonly")

    def _show_placeholder(self):
        if self._placeholder_shown:
            return
        blank = Image.new("RGB", self.preview_size, (30, 30, 30))
        self._photo = ImageTk.PhotoImage(blank)
        self.preview_label.configure(image=self._photo)
        self._placeholder_shown = True
        self.fps_var.set("")

    def _on_preview_resize(self, event):
        width = max(PREVIEW_MIN_SIZE[0], min(PREVIEW_MAX_SIZE[0], event.width - 4))
        height = max(PREVIEW_MIN_SIZE[1], min(PREVIEW_MAX_SIZE[1], event.height - 4))
        if abs(width - self.preview_size[0]) < 24 and abs(height - self.preview_size[1]) < 24:
            return
        self.preview_size = (width, height)
        self.worker.set_preview_size(self.preview_size)
        if not self.camera_open:
            self._placeholder_shown = False
            self._show_placeholder()

    # -- Boucle unique d'affichage -----------------------------------------

    def _tick(self):
        try:
            self._drain_events()
            self._draw_preview()
        finally:
            self.root.after(33, self._tick)

    def _drain_events(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            handler = getattr(self, "_ev_" + kind, None)
            if handler is not None:
                handler(**payload)

    def _draw_preview(self):
        frame, shape, fps = self.worker.take_preview()
        if frame is None:
            return
        image = Image.fromarray(frame)
        # PhotoImage.paste reutilise le buffer Tk existant : pas de nouvelle
        # allocation a chaque image, donc pas de scintillement ni de pression
        # sur le ramasse-miettes.
        if self._photo is not None and (self._photo.width(), self._photo.height()) == image.size:
            self._photo.paste(image)
        else:
            self._photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._photo)
        self._placeholder_shown = False
        if shape:
            self.fps_var.set("{}x{} - {:.1f} img/s".format(shape[0], shape[1], fps))

    # -- Evenements venant du thread camera --------------------------------

    def _ev_status(self, text):
        self.status_var.set(text)

    def _ev_error(self, message):
        self.status_var.set("Erreur : {}".format(message))
        messagebox.showerror("Erreur", message)

    def _ev_busy(self, value):
        self.busy = bool(value)
        self._update_buttons()

    def _ev_state(self, streaming, device):
        self.camera_open = bool(streaming)
        self.device_open = bool(device)
        self._update_buttons()
        if not streaming:
            self._placeholder_shown = False
            self._show_placeholder()

    def _ev_cameras(self, values):
        self.camera_combo["values"] = values
        if self.camera_var.get() not in values:
            self.camera_var.set(values[0])

    def _ev_captured(self, path, width, height, seconds, note):
        megapixels = (width * height) / 1_000_000
        self.status_var.set(
            "Photo enregistree : {} - {}x{} (~{:.1f} MP) en {:.1f} s{}".format(
                path, width, height, megapixels, seconds, note)
        )

    def _ev_detected(self, resolutions):
        if resolutions:
            text = "\n".join("{}x{}  (~{:.1f} MP)".format(w, h, (w * h) / 1e6)
                             for w, h in resolutions)
            top_w, top_h = resolutions[-1]
            # Les prereglages deviennent ceux de cette camera : plus moyen de
            # demander un mode qu'elle ne possede pas et qu'elle remplacerait
            # sans le dire.
            self._set_presets(resolutions, select=(top_w, top_h))
            self._on_preset_selected()
            self.status_var.set("Resolutions reellement fournies : {}".format(
                ", ".join("{}x{}".format(w, h) for w, h in resolutions)))
            messagebox.showinfo(
                "Resolutions detectees",
                "Modes que cette camera fournit reellement :\n\n{}\n\n"
                "Maximum : {}x{} (~{:.1f} MP)\n\n"
                "Les boutons de resolution ont ete remplaces par cette liste.".format(
                    text, top_w, top_h, (top_w * top_h) / 1e6))
        else:
            messagebox.showwarning("Resolutions detectees",
                                   "Aucune resolution testee n'a pu fournir d'image.")

    # -- Actions utilisateur -----------------------------------------------

    def _current_index(self):
        """Index camera lu en tete du libelle ("2 - max 8000x6000 (48.0 MP)")."""
        text = self.camera_var.get() or ""
        digits = ""
        for char in text:
            if not char.isdigit():
                break
            digits += char
        try:
            return int(digits)
        except ValueError:
            return 0

    def _target_resolution(self):
        """Resolution demandee, validee. None si la saisie est invalide."""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except (TypeError, ValueError):
            return None
        if not (16 <= width <= 20000 and 16 <= height <= 20000):
            return None
        return width, height

    def _set_presets(self, presets, select=None):
        """(Re)construit la rangee de boutons radio de resolution."""
        for child in self.preset_row.winfo_children():
            child.destroy()
        for column, res in enumerate(presets):
            megapixels = res[0] * res[1] / 1e6
            ttk.Radiobutton(
                self.preset_row, text="{:.1f} MP ({}x{})".format(megapixels, res[0], res[1]),
                variable=self.res_var, value=fmt_res(res), command=self._on_preset_selected,
            ).grid(row=column // 4, column=column % 4, padx=6, pady=4, sticky="w")
        if select is not None:
            self.res_var.set(fmt_res(select))

    def _on_preset_selected(self):
        res = parse_res(self.res_var.get())
        if res is None:
            return
        self.width_var.set(str(res[0]))
        self.height_var.set(str(res[1]))
        self._apply_resolution()

    def _apply_resolution(self, force=False):
        """Applique la resolution a l'apercu, avec anti-rebond.

        Chaque clic sur un bouton radio declenchait auparavant une fermeture
        puis une reouverture immediates de la camera, soit environ une seconde
        de gel de l'interface. Les clics rapproches sont maintenant regroupes
        en une seule reouverture.
        """
        if self._apply_job is not None:
            self.root.after_cancel(self._apply_job)
            self._apply_job = None
        if not self.camera_open and not force:
            return
        self._apply_job = self.root.after(350, self._do_apply_resolution)

    def _do_apply_resolution(self):
        self._apply_job = None
        target = self._target_resolution()
        if target is None:
            self.status_var.set("Largeur/hauteur invalides : entiers attendus entre 16 et 20000.")
            return
        self.worker.send("open", index=self._current_index(),
                         width=target[0], height=target[1], stream=True)

    def _open_preview(self):
        target = self._target_resolution()
        if target is None:
            messagebox.showerror("Erreur", "Largeur/hauteur invalides.")
            return
        # L'apercu est ouvert a la resolution de capture : les modes basse
        # resolution de ces cameras sont un recadrage numerique du capteur, le
        # champ de vision de l'apercu ne correspondrait pas a celui de la photo.
        self.worker.send("open", index=self._current_index(),
                         width=target[0], height=target[1], stream=True)

    def _close_preview(self):
        # Sans cette annulation, un changement de resolution encore en attente
        # rouvre la camera juste apres la fermeture demandee.
        if self._apply_job is not None:
            self.root.after_cancel(self._apply_job)
            self._apply_job = None
        self.worker.send("close")

    def capture_photo(self):
        if self.busy:
            return                       # empeche deux captures concurrentes
        target = self._target_resolution()
        if target is None:
            messagebox.showerror("Erreur", "Largeur/hauteur invalides.")
            return
        if not os.path.isdir(self.save_dir):
            messagebox.showerror("Erreur",
                                 "Dossier de sauvegarde introuvable :\n{}".format(self.save_dir))
            return
        self.status_var.set("Capture en cours...")
        self.worker.send("capture", index=self._current_index(),
                         width=target[0], height=target[1], save_dir=self.save_dir)

    def detect_resolutions(self):
        if self.busy:
            return
        target = self._target_resolution() or DEFAULT_PRESETS[0]
        self.worker.send("detect", index=self._current_index(), restore_res=target)

    def choose_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.save_dir)
        if chosen:
            self.save_dir = chosen
            self.status_var.set("Dossier de sauvegarde : {}".format(self.save_dir))

    def _on_close(self):
        if self._apply_job is not None:
            try:
                self.root.after_cancel(self._apply_job)
            except Exception:                                  # noqa: BLE001
                pass
        self.worker.shutdown()
        self.worker.join(timeout=3.0)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
