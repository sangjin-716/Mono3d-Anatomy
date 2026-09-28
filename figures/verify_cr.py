"""Prints page size, minimum font size, fonts and embedded images of each camera-ready PDF in
reports_rerun/figures/, and writes a 200-dpi preview to reports_rerun/figures/png/.
Requires PyMuPDF (pip install pymupdf).
Run from the repository root: python figures/verify_cr.py [fig1_progression ...]"""
import pymupdf, sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools._release import out_path
S=out_path("figures")
os.makedirs(f"{S}/png", exist_ok=True)
names=sys.argv[1:] or [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{S}/*.pdf"))]
for n in names:
    f=f"{S}/{n}.pdf"; d=pymupdf.open(f); pg=d[0]
    sp=[s for b in pg.get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"].strip()]
    fonts=set(ff[3]+"/"+ff[1] for ff in pg.get_fonts())
    mn=min(s["size"] for s in sp)
    small=[(s["text"],round(s["size"],2)) for s in sp if round(s["size"],2)<6.5]
    imgs=pg.get_images()
    print(f"{n}: page {pg.rect.width:.1f} x {pg.rect.height:.1f} pt; spans {len(sp)}; min font {mn:.2f} pt; <6.5: {small}; images {len(imgs)}; fonts {sorted(fonts)}")
    pg.get_pixmap(dpi=200).save(f"{S}/png/{n}.png")
