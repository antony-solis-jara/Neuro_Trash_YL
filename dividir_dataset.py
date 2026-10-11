import random, shutil
from pathlib import Path

ORIGEN = Path("dataset/Garbage classification")
DESTINO = Path("dataset_yolo")
random.seed(42)

for clase in ORIGEN.iterdir():
    if not clase.is_dir():
        continue
    imgs = [p for p in clase.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(imgs)
    corte = int(len(imgs) * 0.8)
    for subset, lista in (("train", imgs[:corte]), ("val", imgs[corte:])):
        out = DESTINO / subset / clase.name
        out.mkdir(parents=True, exist_ok=True)
        for p in lista:
            shutil.copy(p, out / p.name)
print("Listo")