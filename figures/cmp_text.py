"""Ported from final_camera_ready/tools/figs/cmp_text.py for the public release.
Computation unchanged. Produces a console report comparing two directories of the five figure PDFs:
text spans (multiset, sizes, positions at 0.1 pt) and vector drawings (colour, fill, points at 0.1 pt).
Requires PyMuPDF (pip install pymupdf).
Run from the repository root: python figures/cmp_text.py <candidate_dir> <reference_dir>
e.g. candidate = reports_rerun/figures (make_figs_cr.py output), reference = the figs/ directory of
the camera-ready LaTeX source."""
import pymupdf, sys, collections
A=sys.argv[2]
B=sys.argv[1]
names=["fig1_progression","fig2_headroom","fig3_anatomy","fig4_waterfall","fig4v2_plane_agreement"]
def spans(p):
    d=pymupdf.open(p); out=[]
    for b in d[0].get_text("dict")["blocks"]:
        for l in b.get("lines",[]):
            for s in l["spans"]:
                if s["text"].strip(): out.append((s["text"],round(s["size"],2),tuple(round(v,1) for v in s["bbox"])))
    return out
def drawings(p):
    d=pymupdf.open(p); out=[]
    for dr in d[0].get_drawings():
        pts=[]
        for it in dr["items"]:
            for v in it[1:]:
                if isinstance(v,pymupdf.Point): pts.append((round(v.x,1),round(v.y,1)))
                elif isinstance(v,pymupdf.Rect): pts+= [(round(v.x0,1),round(v.y0,1)),(round(v.x1,1),round(v.y1,1))]
                elif isinstance(v,pymupdf.Quad): pts+= [(round(q.x,1),round(q.y,1)) for q in (v.ul,v.lr)]
        out.append((dr.get("color"),dr.get("fill"),tuple(pts)))
    return out
for n in names:
    a=spans(f"{A}/{n}.pdf"); b=spans(f"{B}/{n}.pdf")
    ta=collections.Counter(s[0] for s in a); tb=collections.Counter(s[0] for s in b)
    print(f"== {n}: {len(a)} spans orig, {len(b)} repro; text multiset identical: {ta==tb}")
    if ta!=tb:
        print("  only orig:", ta-tb); print("  only repro:", tb-ta)
    sa=collections.Counter((s[0],s[1]) for s in a); sb=collections.Counter((s[0],s[1]) for s in b)
    print("  text+size identical:", sa==sb)
    pa=[s for s in a]; pb=[s for s in b]
    posdiff=sum(1 for x,y in zip(sorted(pa),sorted(pb)) if x!=y)
    print("  spans with any bbox diff (0.1pt):", posdiff)
    da=drawings(f"{A}/{n}.pdf"); db=drawings(f"{B}/{n}.pdf")
    print(f"  drawings: {len(da)} vs {len(db)}; identical: {da==db}")
    if da!=db:
        ca=collections.Counter(da); cb=collections.Counter(db)
        print("   n only orig:", sum((ca-cb).values()), " n only repro:", sum((cb-ca).values()))
