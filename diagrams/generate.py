"""
diagrams/generate.py

Regenerates all 7 architecture SVGs in this folder from scratch. Run with
`python generate.py` from anywhere (paths are resolved relative to this
file's own location, not the working directory). Edit a diagram by editing
its function below, then rerun.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svg_helpers import box, group, txt, arrow, elbow, write_svg

OUT = os.path.dirname(os.path.abspath(__file__))


def legend(x, y, items):
    """items: list of (kind, label). Item width is sized from label length so longer labels don't collide with the next swatch."""
    parts = []
    bx = x
    for kind, label in items:
        parts.append(box(bx, y, 20, 20, "", kind=kind, rx=4))
        parts.append(txt(bx + 26, y + 15, label, size=11, anchor="start", color="#374151"))
        bx += 26 + len(label) * 6.6 + 34
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. Overall architecture
# ---------------------------------------------------------------------------
def diagram_overall():
    W, H = 1560, 640
    p = []
    p.append(txt(W/2, 32, "PhysChemCAL — Overall Architecture", size=18, weight="700"))
    p.append(legend(40, 55, [("data", "data / infra"), ("physics", "PhysChem encoder"),
                              ("cal", "CAL head"), ("output", "inference output"), ("explain", "Phase-3 (optional)")]))

    y = 190; h = 80
    b1 = (40, y, 220, h)
    b2 = (330, y, 220, h)
    b3 = (620, y, 280, h)
    b4 = (970, y, 240, h)

    p.append(box(*b1, "Cached .pkl record", "mol (3D-embedded)\n+ smiles + label", kind="data", ))
    p.append(box(*b2, "Featurize + Batch", "atom/bond feats\n+ MaskMatrices  (fig. 2)", kind="data"))
    p.append(box(*b3, "PhysChem Encoder", "Initializer → (PhysNet↔ChemNet)×2\n(fig. 3-5)", kind="physics"))
    p.append(box(*b4, "CAL Head", "attn split → xc/xo → pool\n→ 3 heads  (fig. 6)", kind="cal"))

    p.append(arrow(b1[0]+b1[2], y+h/2, b2[0], y+h/2))
    p.append(arrow(b2[0]+b2[2], y+h/2, b3[0], y+h/2))
    p.append(arrow(b3[0]+b3[2], y+h/2, b4[0], y+h/2, label="hv [A,200]", label_dy=-8))

    # outputs
    out_x = 1270
    o1 = (out_x, 110, 250, 60)
    o2 = (out_x, 340, 250, 70)
    p.append(box(*o1, "o_pred  →  pEC50", "inference output (only branch used)", kind="output"))
    p.append(box(*o2, "c_pred, co_pred", "→ Losses (training only)", kind="data", dashed=True))
    p.append(elbow([(b4[0]+b4[2], y+20), (out_x-20, y+20), (out_x-20, 140), (out_x, 140)], label="o_pred, att_o"))
    p.append(elbow([(b4[0]+b4[2], y+60), (out_x-20, y+60), (out_x-20, 375), (out_x, 375)], label="c_pred, co_pred", dashed=True))

    # conformations -> losses (routed below o2, not through it)
    p.append(elbow([(b3[0]+80, y+h), (b3[0]+80, 460), (out_x+125, 460), (out_x+125, o2[1]+o2[3])],
                    label="conformations → conf_loss", dashed=True, label_dx=-60, label_dy=14))

    # phase 3 branch -- fed straight down from CAL Head, clear of the outputs column entirely
    ph = (970, 500, 260, 90)
    p.append(box(*ph, "Phase-3 Counterfactual Explainer", "only runs with --phase3 (fig. 7)\nexpensive, off by default", kind="explain", dashed=True))
    p.append(elbow([(b4[0]+b4[2]/2, y+h), (b4[0]+b4[2]/2, ph[1])],
                    label="o_pred, att_o  (only if --phase3)", dashed=True, key=True, label_dy=16))

    write_svg(f"{OUT}/01_overall_architecture.svg", W, H,
              "PhysChemCAL overall architecture: data to prediction, with optional Phase-3 branch",
              p, caption="Solid = always runs. Dashed = training-only or --phase3-only. See figures 2-7 for module internals.")


# ---------------------------------------------------------------------------
# 2. Featurisation + batching
# ---------------------------------------------------------------------------
def diagram_featurization():
    W, H = 1300, 620
    p = []
    p.append(txt(W/2, 32, "Featurisation + Batching (src/data/mask_matrices.py)", size=18, weight="700"))

    mols = [("mol 1 (A=20)", "local idx 0..19", 60), ("mol 2 (A=18)", "local idx 0..17", 280), ("mol 3 (A=22)", "local idx 0..21", 500)]
    for label, sub, x in mols:
        p.append(box(x, 80, 200, 60, label, sub, kind="data"))
        p.append(arrow(x+100, 140, 350, 230, dashed=False))

    bb = (250, 230, 300, 60)
    p.append(box(*bb, "batch_graphs()", "concat, offset begin_idx/end_idx by cumulative atom count", kind="key"))
    p.append(arrow(bb[0]+bb[2]/2, bb[1]+bb[3], bb[0]+bb[2]/2, 340))

    rb = (100, 340, 480, 70)
    p.append(box(*rb, "Flat batch (A=60 atoms total)", "mol2 begin_idx += 20    mol3 begin_idx += 38", kind="data"))

    # block diagonal matrix visual
    mx, my, cell = 750, 90, 90
    p.append(txt(mx + 1.5*cell, my - 16, "vertex_edge_w1 [A,E] — block-diagonal", size=13, weight="700"))
    sizes = [20, 18, 22]
    labels = ["mol1", "mol2", "mol3"]
    y0 = my
    for i in range(3):
        x0 = mx
        for j in range(3):
            if i == j:
                p.append(box(x0, y0, cell, cell, labels[i], f"{sizes[i]}x{sizes[i]}", kind="chem", rx=4))
            else:
                p.append(box(x0, y0, cell, cell, "0", kind="data", rx=4))
            x0 += cell + 6
        y0 += cell + 6
    p.append(elbow([(rb[0]+rb[2], rb[1]+35), (mx - 30, rb[1]+35), (mx-30, my+1.5*cell), (mx, my+1.5*cell)],
                    label="same offsets build every mask", key=True, label_dx=-10))
    p.append(txt(mx + 1.5*cell, my + 3*cell + 35, "off-diagonal blocks are always 0 —", size=11.5, color="#6b7280", italic=True))
    p.append(txt(mx + 1.5*cell, my + 3*cell + 52, "a bond in mol 2 can never attend into mol 3", size=11.5, color="#6b7280", italic=True))

    write_svg(f"{OUT}/02_featurization.svg", W, H,
              "Featurisation and batching: per-molecule graphs concatenated into one flat, block-diagonal batch",
              p, caption="Every mask/index tensor (MaskMatrices, edge_index, batch) is built from this same offset, so they stay consistent by construction.")


# ---------------------------------------------------------------------------
# 3. Initializer
# ---------------------------------------------------------------------------
def diagram_initializer():
    W, H = 1300, 420
    p = []
    p.append(txt(W/2, 32, "Initializer (src/models/initializer.py)", size=18, weight="700"))
    p.append(legend(40, 50, [("data", "tensor"), ("key", "learned")]))

    yA = 110
    a1 = (40, yA, 160, 60); a2 = (250, yA, 180, 60); a3 = (480, yA, 220, 60); a4 = (750, yA, 220, 60); a5 = (1020, yA, 220, 60)
    p.append(box(*a1, "atom_ftr", "[A, 34]", kind="data"))
    p.append(box(*a2, "Linear + Tanh", "34 → 200", kind="data"))
    p.append(box(*a3, "Residual GCN × 2", "concat, not replace", kind="data"))
    p.append(box(*a4, "2-layer LSTM", "per-molecule sequence", kind="data"))
    p.append(box(*a5, "p [A, 3]", "learned initial momentum", kind="key"))
    for b1, b2, lbl in [(a1, a2, None), (a2, a3, "hv0[A,200]"), (a3, a4, "hv[A,200]"), (a4, a5, None)]:
        p.append(arrow(b1[0]+b1[2], b1[1]+30, b2[0], b2[1]+30, label=lbl))

    yB = 300
    b1 = (40, yB, 160, 60); b2 = (1020, yB, 220, 60)
    p.append(box(*b1, "pos", "[A, 3]  real 3D coords", kind="data"))
    p.append(box(*b2, "q [A, 3]", "= pos, unchanged here", kind="key"))
    p.append(arrow(b1[0]+b1[2], b1[1]+30, b2[0], b2[1]+30, label="no learning on this path — q starts from real geometry", label_dy=16, key=True))

    write_svg(f"{OUT}/03_initializer.svg", W, H,
              "Initializer: atom features become the learned momentum p, while position q is copied directly from the real conformer",
              p, caption="p (\"which way would this atom tend to move\") is learned; q (\"where the atom actually is\") is not — PhysNet must start from real geometry, not a guess.")


# ---------------------------------------------------------------------------
# 4. PhysNet
# ---------------------------------------------------------------------------
def diagram_physnet():
    W, H = 1360, 640
    p = []
    p.append(txt(W/2, 32, "PhysNet — one Newton step (src/models/physnet.py)", size=18, weight="700"))
    p.append(legend(40, 50, [("physics", "force computation"), ("key", "the O(A^2)→O(chunk x A) fix")]))

    # bond force (left)
    p.append(group(40, 100, 560, 200, "Bond force", kind="physics"))
    bf1 = (70, 130, 220, 55); bf2 = (330, 130, 240, 55); bf3 = (170, 220, 340, 55)
    p.append(box(*bf1, "hv[begin], hv[end], he", kind="data"))
    p.append(box(*bf2, "BondForceMLP", "final layer zero-init", kind="physics"))
    p.append(box(*bf3, "magnitude · unit_dir(q)", "→ index_add_ by begin_idx → f_bond[A,3]", kind="physics"))
    p.append(arrow(bf1[0]+bf1[2], bf1[1]+27, bf2[0], bf2[1]+27))
    p.append(arrow(bf2[0]+bf2[2]/2, bf2[1]+55, bf3[0]+bf3[2]/2+80, bf3[1]))

    # relational force (right, KEY)
    p.append(group(640, 100, 680, 200, "Relational force — CHUNKED", kind="key"))
    ry = 130
    for i, lbl in enumerate(["chunk 1 [32,A]", "chunk 2 [32,A]", "... chunk n"]):
        p.append(box(660, ry + i*38, 170, 32, lbl, kind="key"))
    rf2 = (880, 150, 260, 60)
    p.append(box(*rf2, "RelationalForceMLP", "per chunk — never materialises full [A,A]", kind="key"))
    p.append(arrow(830, 165, rf2[0], rf2[1]+30))
    rf3 = (1180, 150, 120, 60)
    p.append(box(*rf3, "f_rela[A,3]", "Σ over A", kind="key"))
    p.append(arrow(rf2[0]+rf2[2], rf2[1]+30, rf3[0], rf3[1]+30))

    # converge
    tot = (560, 340, 260, 55)
    p.append(box(*tot, "f_total = f_bond + f_rela", kind="data"))
    p.append(elbow([(bf3[0]+bf3[2]/2+80, bf3[1]+55), (bf3[0]+bf3[2]/2+80, 367), (tot[0], 367)]))
    p.append(elbow([(rf3[0]+rf3[2]/2, rf3[1]+60), (rf3[0]+rf3[2]/2, 367), (tot[0]+tot[2], 367)]))

    upd = (480, 430, 420, 70)
    p.append(box(*upd, "Newton update", "p += f_total·τ    q += (p/m)·τ    (τ=0.25)", kind="physics"))
    p.append(arrow(tot[0]+tot[2]/2, tot[1]+55, upd[0]+upd[2]/2, upd[1]))

    conf = (480, 550, 420, 55)
    p.append(box(*conf, "snapshot q → conformations[]", "used later by conf_loss", kind="output"))
    p.append(arrow(upd[0]+upd[2]/2, upd[1]+70, conf[0]+conf[2]/2, conf[1]))
    p.append(txt(upd[0]+upd[2]+40, upd[1]+35, "loop × N_ITERATION=2\nper PhysNet call", size=11.5, anchor="start", color="#6b7280", italic=True))

    write_svg(f"{OUT}/04_physnet.svg", W, H,
              "PhysNet: learned bond-spring and chunked relational forces combine into a Newtonian position update",
              p, caption="The chunked relational force (amber) is the memory fix from report Table 3: peak tensor drops from [A,A,feat] to [32,A,feat], ~20x smaller.")


# ---------------------------------------------------------------------------
# 5. ChemNet
# ---------------------------------------------------------------------------
def diagram_chemnet():
    W, H = 1360, 600
    p = []
    p.append(txt(W/2, 32, "ChemNet — triplet attention for one hub atom (src/models/chemnet.py)", size=18, weight="700"))

    # geometry panel
    gx, gy, gw, gh = 40, 90, 340, 340
    p.append(box(gx, gy, gw, gh, "", kind="data", rx=10))
    Ax, Ay = gx+gw/2, gy+gh/2+20
    Bx, By = gx+70, gy+90
    Cx, Cy = gx+gw-70, gy+100
    p.append(f'<line x1="{Ax}" y1="{Ay}" x2="{Bx}" y2="{By}" stroke="#7e22ce" stroke-width="2"/>')
    p.append(f'<line x1="{Ax}" y1="{Ay}" x2="{Cx}" y2="{Cy}" stroke="#7e22ce" stroke-width="2"/>')
    p.append(f'<path d="M {Ax-40},{Ay-15} A 45,45 0 0 1 {Ax+40},{Ay-18}" fill="none" stroke="#b45309" stroke-width="1.5"/>')
    p.append(txt(Ax, Ay-55, "angle BAC", size=11, color="#b45309"))
    for (cx, cy, lbl, sub) in [(Ax, Ay, "A", "hub"), (Bx, By, "B", "neighbour"), (Cx, Cy, "C", "neighbour")]:
        p.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="#e9d5ff" stroke="#7e22ce" stroke-width="2"/>')
        p.append(txt(cx, cy - 16, lbl, size=13, weight="700"))
        p.append(txt(cx, cy + 26, sub, size=10.5, color="#6b7280"))
    p.append(txt(gx+gw/2, gy+gh+22, "degree-k atom → k neighbours → k×k pairs", size=11.5, color="#6b7280", italic=True))

    # pipeline
    x0, w0 = 430, 570
    steps = [
        ("unit_dir(A→B), unit_dir(A→C) + dist(A,B)", None, "chem"),
        ("cos(angle BAC) = dot(dirAB, dirAC)", None, "chem"),
        ("concat[ hv_B | dist | hv_A | hv_C | angle ]", None, "chem"),
        ("message_mlp  →  pair_msg [k×k, 200]", None, "chem"),
        ("edge-attn weight × max-pool over k×k pairs  →  mv[A]", "KEY: this is the aggregation, not a simple sum", "key"),
        ("GRU(mv[A], hv[A])  →  hv_new[A]", None, "chem"),
    ]
    y = 90
    prev = None
    for label, sub, kind in steps:
        b = (x0, y, w0, 60)
        p.append(box(*b, label, sub, kind=kind))
        if prev:
            p.append(arrow(x0+w0/2, prev[1]+prev[3], x0+w0/2, b[1], key=(kind == "key")))
        prev = b
        y += 82

    write_svg(f"{OUT}/05_chemnet.svg", W, H,
              "ChemNet triplet attention: bond angles at one hub atom become part of its updated hidden state",
              p, caption="Grouped by source atom (not a global [2E,2E] tensor) — every atom's k~4 bonds cost O(k^2), so the whole batch costs O(E·k^2), not O(E^2).")


# ---------------------------------------------------------------------------
# 6. CAL head
# ---------------------------------------------------------------------------
def diagram_cal():
    W, H = 1360, 780
    p = []
    p.append(txt(W/2, 32, "CAL Head (src/models/cal_head.py)", size=18, weight="700"))
    p.append(legend(40, 50, [("cal", "context branch"), ("key", "causal branch (inference)"), ("data", "training-only")]))

    hv = (560, 90, 240, 55)
    p.append(box(*hv, "hv [A, 200]", kind="data"))
    att = (460, 175, 440, 55)
    p.append(box(*att, "node_att_mlp + softmax  →  att_c, att_o", kind="cal"))
    p.append(arrow(hv[0]+hv[2]/2, hv[1]+55, att[0]+att[2]/2, att[1]))

    def branch(x, kind, prefix, final_label, final_kind):
        rows = [
            (f"{prefix} · hv", kind),
            ("proj (Linear+ReLU) → mean-pool(mol_vertex_w)", kind),
            ("LayerNorm → x_mol [M,200]", kind),
            ("branch_head (2-layer MLP)", kind),
            (final_label, final_kind),
        ]
        y = 280
        boxes = []
        prev = None
        for label, k in rows:
            b = (x, y, 260, 55)
            p.append(box(*b, label, kind=k))
            if prev:
                p.append(arrow(x+130, prev[1]+55, x+130, b[1], key=(k == "key")))
            prev = b
            boxes.append(b)
            y += 78
        return boxes

    left = branch(60, "cal", "att_c", "c_pred  (Var → 0, training)", "data")
    right = branch(1040, "key", "att_o", "o_pred → pEC50 (inference)", "output")

    p.append(elbow([(att[0], att[1]+27), (190, att[1]+27), (190, 280)]))
    p.append(elbow([(att[0]+att[2], att[1]+27), (1170, att[1]+27), (1170, 280)], key=True))

    # backdoor / shuffle path
    sh = (500, 500, 320, 55)
    p.append(box(*sh, "shuffle xc_mol across batch  +  .detach()", "no gradient flows back into context branch", kind="key"))
    p.append(elbow([(left[2][0]+left[2][2], left[2][1]+27), (sh[0], sh[1]+27)], label="xc_mol", key=True))

    mg = (500, 578, 320, 55)
    p.append(box(*mg, "xo_mol + xc_shuffled", kind="key"))
    p.append(arrow(sh[0]+sh[2]/2, sh[1]+55, mg[0]+mg[2]/2, mg[1]))
    p.append(elbow([(right[2][0], right[2][1]+27), (mg[0]+mg[2], mg[1]+27)], label="xo_mol", key=True))

    ch = (500, 656, 320, 30)
    p.append(box(500, 656, 320, 36, "combined_head → co_pred (training only)", kind="data", dashed=True))
    p.append(arrow(mg[0]+mg[2]/2, mg[1]+55, mg[0]+mg[2]/2, 656))

    write_svg(f"{OUT}/06_cal_head.svg", W, H,
              "CAL head: one attention split feeds a context branch, a causal branch, and a shuffled backdoor-adjustment branch",
              p, caption="The shuffle+detach step (amber) is the backdoor adjustment: it forces o_pred to stay accurate even when the scaffold is swapped for a random one from the batch.")


# ---------------------------------------------------------------------------
# 7. Phase-3 counterfactual explainer
# ---------------------------------------------------------------------------
def diagram_phase3():
    W, H = 1500, 520
    p = []
    p.append(txt(W/2, 32, "Phase-3 Counterfactual Explainer (src/explain/counterfactual.py)", size=18, weight="700"))

    q = (40, 80, 200, 55)
    p.append(box(*q, "query SMILES", kind="data"))
    pa = (40, 165, 200, 60)
    p.append(box(*pa, "predict_with_attention()", "1 model call", kind="key"))
    p.append(arrow(q[0]+q[2]/2, q[1]+55, pa[0]+pa[2]/2, pa[1]))
    ca = (40, 255, 200, 60)
    p.append(box(*ca, "causal_atoms", "{i : att_o[i] > θ}", kind="data"))
    p.append(arrow(pa[0]+pa[2]/2, pa[1]+60, ca[0]+ca[2]/2, ca[1]))

    st = (300, 80, 220, 55)
    p.append(box(*st, "STONED mutate × 500", "selfies token edits", kind="data"))
    p.append(arrow(q[0]+q[2], q[1]+27, st[0], st[1]+27))

    mcs = (570, 80, 220, 55)
    p.append(box(*mcs, "MCS filter vs causal_atoms", "RDKit, 2D only", kind="key"))
    p.append(arrow(st[0]+st[2], st[1]+27, mcs[0], mcs[1]+27))
    p.append(elbow([(ca[0]+ca[2], ca[1]+30), (mcs[0]+mcs[2]/2, ca[1]+30), (mcs[0]+mcs[2]/2, mcs[1]+55)], key=True))

    divider_x = 850
    p.append(f'<line x1="{divider_x}" y1="60" x2="{divider_x}" y2="480" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="4,4"/>')
    p.append(txt(divider_x-10, 55, "CPU only, no model calls", size=11, anchor="end", color="#6b7280", italic=True))
    p.append(txt(divider_x+10, 55, "GPU model calls", size=11, anchor="start", color="#6b7280", italic=True))

    sc = (880, 80, 300, 70)
    p.append(box(*sc, "survivors (~200)", "3D embed + batched predict_smiles()\nONE model call, not 200", kind="key"))
    p.append(arrow(mcs[0]+mcs[2], mcs[1]+27, sc[0], sc[1]+35))

    sim = (890, 190, 260, 55)
    p.append(box(*sim, "Tanimoto sim + split by Δpred (up/down)", kind="data"))
    p.append(arrow(sc[0]+sc[2]/2, sc[1]+70, sim[0]+sim[2]/2, sim[1]))

    db = (890, 285, 260, 55)
    p.append(box(*db, "DBSCAN cluster (eps=0.15)", "→ top-2 diverse per direction", kind="data"))
    p.append(arrow(sim[0]+sim[2]/2, sim[1]+55, db[0]+db[2]/2, db[1]))

    out = (890, 380, 260, 60)
    p.append(box(*out, "counterfactuals[] + alignment_score", kind="output"))
    p.append(arrow(db[0]+db[2]/2, db[1]+55, out[0]+out[2]/2, out[1]))

    write_svg(f"{OUT}/07_phase3_counterfactual.svg", W, H,
              "Phase-3 counterfactual explainer: cheap CPU filtering happens before any expensive GPU scoring",
              p, caption="The MCS filter runs before 3D embedding, on the left of the CPU/GPU boundary — so the ~60% of candidates that fail it never pay the embed+model cost.")


if __name__ == "__main__":
    diagram_overall()
    diagram_featurization()
    diagram_initializer()
    diagram_physnet()
    diagram_chemnet()
    diagram_cal()
    diagram_phase3()
