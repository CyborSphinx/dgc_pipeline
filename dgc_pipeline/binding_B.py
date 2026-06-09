"""
binding_B.py — build a binding-based TF->gene influence matrix B (mouse, mm10)
==============================================================================
Faithful mouse analog of DGC's B matrix:

    B[i, m] = a_m * s_i * hits(m, i)

  hits(m,i) : # of TF-m binding events in gene i's promoter [TSS-w, TSS+w]
              -- from ChIP-seq peaks where available (OSKM), else JASPAR motif scan
  s_i       : accessibility of gene i's promoter (1 if overlaps an ATAC/H3K27ac
              peak, else 0)  -- DGC's DNase analog
  a_m       : +1 activator / -1 repressor

Inputs (all local files you already have):
  - refGene.txt(.gz)        : UCSC mm10 genePred+name2  -> per-gene TSS
  - GSE99009 peak BEDs       : ChIP (Oct4/Sox2/Klf4 ...) + ATAC/H3K27ac
  - JASPAR PFM text          : motifs for the non-ChIP TFs
  - mm10.fa(.gz) [optional]  : only needed for JASPAR motif scanning

Design notes / honesty:
  * ChIP peaks are MEASURED binding -> strictly better than motif prediction.
    For OSKM we use ChIP; for everyone else we fall back to JASPAR scans.
  * If mm10.fa is not provided, motif scanning is skipped and only ChIP-backed
    TFs get nonzero columns (clearly reported). No fabrication.
  * Accessibility uses ATAC and/or H3K27ac peaks (active marks). Repressive
    marks (H3K9me3/H3K27me3) and siUbc9 perturbation files are IGNORED.
"""

import gzip
import glob
import os
import re
import numpy as np


# ---- which GSE99009 files to USE vs IGNORE (wild-type, active only) ----
CHIP_TF_PATTERNS = {           # map filename token -> TF symbol
    "Oct4": "Pou5f1", "Sox2": "Sox2", "Klf4": "Klf4", "Myc": "Myc",
}
ACCESS_TOKENS = ("ATAC", "H3K27Ac", "H3K27ac", "H3K4Me1", "H3K4me1",
                 "H3K4Me3", "H3K4me3")          # active / open marks
IGNORE_TOKENS = ("siUbc9", "shUbc9", "H3K9me3", "H3K9Me3",
                 "H3K27Me3", "H3K27me3", "SUMO")  # wrong arm / repressive
# NOTE: do NOT put bare "Ubc9" here -- the MEF ATAC files are named
# "ATAC-seq_MEF_Ubc9-..." and are legitimate accessibility data. Only the
# sh/si-Ubc9 KNOCKDOWN files are the perturbation arm to exclude.


# ============================================================
# TSS from refGene
# ============================================================
def _open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_tss(refgene_path, valid_chroms=None):
    """
    Parse UCSC refGene (genePred + name2). Returns dict symbol -> (chrom, tss, strand).
    If a gene has multiple isoforms, keep the most upstream TSS on its strand
    (a simple, defensible default).
    cols (0-based): 2 chrom, 3 strand, 4 txStart, 5 txEnd, 12 name2
    """
    best = {}
    with _open_maybe_gz(refgene_path) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 13:
                continue
            chrom, strand = p[2], p[3]
            if "_" in chrom:                      # drop random/alt scaffolds
                continue
            if valid_chroms and chrom not in valid_chroms:
                continue
            try:
                tx_start, tx_end = int(p[4]), int(p[5])
            except ValueError:
                continue
            sym = p[12]
            tss = tx_start if strand == "+" else tx_end
            if sym not in best:
                best[sym] = (chrom, tss, strand, tx_start, tx_end)
            else:
                # keep most upstream TSS; widen body extent to cover all isoforms
                c0, t0, s0, b0s, b0e = best[sym]
                bs, be = min(b0s, tx_start), max(b0e, tx_end)
                if strand == "+" and tss < t0:
                    best[sym] = (chrom, tss, strand, bs, be)
                elif strand == "-" and tss > t0:
                    best[sym] = (chrom, tss, strand, bs, be)
                else:
                    best[sym] = (chrom, t0, s0, bs, be)
    print(f"  TSS: {len(best)} gene symbols from refGene")
    return best


def promoter_intervals(tss_map, genes, window=5000, include_body=False):
    """
    For each gene, return (chrom, start, end). If include_body, the interval is
    [txStart - window, txEnd + window] (promoter + gene body + flank, captures
    enhancers within `window`); else the TSS-centered [tss-window, tss+window].
    """
    out = {}
    for g in genes:
        rec = tss_map.get(g)
        if rec is None:
            out[g] = None
            continue
        chrom, tss, strand, b_s, b_e = rec
        if include_body:
            out[g] = (chrom, max(0, b_s - window), b_e + window)
        else:
            out[g] = (chrom, max(0, tss - window), tss + window)
    n_have = sum(1 for v in out.values() if v)
    print(f"  intervals: {n_have}/{len(genes)} genes "
          f"({'body+/-' if include_body else 'TSS+/-'}{window}bp)")
    return out


# ============================================================
# BED peak loading + interval overlap
# ============================================================
def load_bed_peaks(path):
    """Return dict chrom -> sorted np.array of [start,end] peak intervals."""
    by = {}
    op = _open_maybe_gz(path) if path.endswith(".gz") else open(path)
    with op as f:
        for ln in f:
            if ln.startswith(("#", "track", "browser")):
                continue
            p = ln.split("\t")
            if len(p) < 3:
                continue
            c = p[0]
            try:
                s, e = int(p[1]), int(p[2])
            except ValueError:
                continue
            by.setdefault(c, []).append((s, e))
    for c in by:
        by[c] = np.array(sorted(by[c]), dtype=np.int64)
    return by


def count_overlaps(promoter, peaks_by_chrom):
    """Count peaks overlapping a (chrom,start,end) promoter. Binary-search starts."""
    if promoter is None:
        return 0
    c, s, e = promoter
    arr = peaks_by_chrom.get(c)
    if arr is None or len(arr) == 0:
        return 0
    # peaks with start <= e
    lo = np.searchsorted(arr[:, 0], e, side="right")
    if lo == 0:
        return 0
    cand = arr[:lo]
    # of those, end >= s   -> overlap
    return int(np.sum(cand[:, 1] >= s))


def any_overlap(promoter, peaks_by_chrom):
    return count_overlaps(promoter, peaks_by_chrom) > 0


# ============================================================
# JASPAR motif scanning (only if genome provided)
# ============================================================
def parse_jaspar(jaspar_path):
    """
    Parse a motif file, AUTO-DETECTING format:
      * MEME format  ('MEME version' header; blocks 'MOTIF <id> <name>' then
        'letter-probability matrix:' then W rows x 4 prob cols)
      * raw JASPAR   ('>MAxxxx.v NAME' then 4 rows x W count cols)
    Returns dict keyed by UPPERCASED motif name(s) -> PFM as (4 x W), rows ACGT.
    Dimer names A::B are also indexed under each component.
    """
    with open(jaspar_path) as f:
        head = f.read(200)
    if "MEME version" in head:
        return _parse_meme(jaspar_path)
    return _parse_jaspar_raw(jaspar_path)


def _store_pfm(pfms, nm, mat_4xw):
    if nm is None or mat_4xw is None or mat_4xw.shape[0] != 4 or mat_4xw.shape[1] == 0:
        return
    pfms[nm.upper()] = mat_4xw
    if "::" in nm:
        for part in nm.split("::"):
            p = part.strip().upper()
            if p and p not in pfms:
                pfms[p] = mat_4xw


def _parse_meme(path):
    """MEME minimal format -> dict name -> (4 x W) PFM (rows A,C,G,T)."""
    pfms = {}
    name = None
    rows = []
    reading = False
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("MOTIF"):
                if name and rows:
                    _store_pfm(pfms, name, np.array(rows, float).T)
                parts = s.split()
                if len(parts) >= 3:
                    name = parts[2]
                elif len(parts) == 2:
                    name = parts[1]
                else:
                    name = None
                if name:
                    name = name.split("(")[0].strip()
                rows = []
                reading = False
            elif s.startswith("letter-probability"):
                reading = True
            elif reading:
                nums = re.findall(r"[-\d.eE+]+", s)
                if len(nums) >= 4:
                    try:
                        rows.append([float(x) for x in nums[:4]])
                    except ValueError:
                        reading = False
                elif s == "" or s.startswith("URL"):
                    reading = False
        if name and rows:
            _store_pfm(pfms, name, np.array(rows, float).T)
    print(f"  MEME: {len(pfms)} motif name-keys parsed")
    return pfms


def _parse_jaspar_raw(path):
    """raw JASPAR '>MAxxxx.v NAME' + 4 count rows -> dict name -> (4 x W)."""
    pfms = {}
    name, rows = None, []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith(">"):
                if name and len(rows) == 4:
                    _store_pfm(pfms, name, np.array(rows, float))
                parts = ln[1:].split()
                if len(parts) >= 2 and re.match(r"^MA\d+", parts[0]):
                    name = " ".join(parts[1:])
                else:
                    name = parts[0] if parts else None
                if name:
                    name = name.split("(")[0].strip()
                rows = []
            else:
                nums = re.findall(r"[-\d.]+", ln)
                if nums:
                    rows.append([float(x) for x in nums])
        if name and len(rows) == 4:
            _store_pfm(pfms, name, np.array(rows, float))
    print(f"  JASPAR(raw): {len(pfms)} motif name-keys parsed")
    return pfms

def pfm_to_logodds(pfm, bg=0.25, pseudo=0.8):
    """
    Position weights (log-odds vs uniform bg). Handles BOTH inputs:
      * counts (raw JASPAR): columns sum to nsites -> add pseudocount, normalize
      * probabilities (MEME): columns already sum to ~1 -> light pseudocount
    Detected by whether column sums are ~1.
    """
    colsums = pfm.sum(axis=0)
    is_prob = np.allclose(colsums, 1.0, atol=0.05)
    if is_prob:
        ppm = (pfm + 1e-3) / (1.0 + 4e-3)         # tiny floor to avoid log(0)
    else:
        ppm = (pfm + pseudo) / (colsums[None, :] + 4 * pseudo)
    return np.log2(ppm / bg)


_B2I = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def _seq_to_idx(seq):
    return np.array([_B2I.get(b, -1) for b in seq], dtype=np.int64)


def scan_motif(seq_idx, lo_fwd, lo_rev, threshold):
    """
    Count positions (both strands) where log-odds score >= threshold.
    VECTORIZED: one-hot the sequence, then every window score is a single
    correlation. ~50-100x faster than the per-position Python loop.
    """
    w = lo_fwd.shape[1]
    L = len(seq_idx)
    if L < w:
        return 0
    valid = seq_idx >= 0
    # one-hot (4 x L); invalid bases -> all-zero column (contributes 0, then masked)
    oh = np.zeros((4, L))
    vi = np.where(valid)[0]
    oh[seq_idx[vi], vi] = 1.0
    # window score for motif position p, start s = sum_p lo[base(s+p), p]
    # = sum over rows of (lo_fwd * oh_window). Use sliding via correlate per row.
    # Build scores by convolving each of 4 base-rows.
    sf = np.zeros(L - w + 1)
    sr = np.zeros(L - w + 1)
    for b in range(4):
        # contribution of base b at motif-position p is lo_fwd[b,p]; slide over oh[b]
        sf += np.convolve(oh[b][::-1], lo_fwd[b][::-1], mode="valid")[::-1] \
            if False else _slide_dot(oh[b], lo_fwd[b])
        sr += _slide_dot(oh[b], lo_rev[b])
    # mask windows containing any invalid base
    if not valid.all():
        bad = ~valid
        badcum = np.convolve(bad.astype(int), np.ones(w, int), mode="valid")
        good = badcum == 0
        sf = np.where(good, sf, -np.inf)
        sr = np.where(good, sr, -np.inf)
    return int(np.sum(sf >= threshold) + np.sum(sr >= threshold))


def _slide_dot(signal, kernel):
    """Sliding dot product: out[s] = sum_p signal[s+p]*kernel[p]. len = L-w+1."""
    w = len(kernel)
    L = len(signal)
    if L < w:
        return np.zeros(0)
    # use as_strided-free approach via correlate
    return np.correlate(signal, kernel, mode="valid")


def load_genome(fasta_path):
    """Load mm10 fasta into dict chrom -> sequence string. Memory-heavy (~2.6GB)."""
    try:
        from pyfaidx import Fasta
        print("  genome: using pyfaidx (indexed, low memory)")
        return ("faidx", Fasta(fasta_path))
    except Exception:
        print("  genome: pyfaidx unavailable; loading FASTA into memory")
        seqs, chrom, buf = {}, None, []
        op = _open_maybe_gz(fasta_path) if fasta_path.endswith(".gz") else open(fasta_path)
        with op as f:
            for ln in f:
                if ln.startswith(">"):
                    if chrom:
                        seqs[chrom] = "".join(buf)
                    chrom = ln[1:].split()[0]
                    buf = []
                else:
                    buf.append(ln.strip())
            if chrom:
                seqs[chrom] = "".join(buf)
        return ("dict", seqs)


def get_seq(genome, chrom, start, end):
    kind, obj = genome
    if kind == "faidx":
        try:
            return str(obj[chrom][start:end])
        except Exception:
            return ""
    return obj.get(chrom, "")[start:end]


# ============================================================
# ACTIVATOR / REPRESSOR sign
# ============================================================
# Minimal defensible map for the TFs in play; extend as needed.
# ### CHOICE ### source from a curated table for a publishable run.
DEFAULT_ACTIVITY = {
    "Pou5f1": +1, "Sox2": +1, "Klf4": +1, "Myc": +1, "Nanog": +1,
    "Esrrb": +1, "Zfp42": +1, "Obox6": +1, "Sall4": +1, "Lin28a": +1,
    "Nr5a2": +1, "Prdm14": +1, "Tfcp2l1": +1, "Utf1": +1,
    "Dppa2": +1, "Dppa4": +1, "Dppa5a": +1,
}


# ============================================================
# MAIN BUILDER
# ============================================================
def build_B_binding(genes, tf_list, refgene_path, gse_dir,
                    jaspar_path=None, genome_path=None,
                    window=5000, motif_threshold=8.0,
                    activity_map=None, verbose=True, restrict_tfs=None,
                    include_body=False):
    """
    Returns (B, tfs_present) where B is (len(genes) x len(tfs_present)).

    restrict_tfs: optional set/list of TF symbols. If given, motif scanning is
    done ONLY for these TFs (e.g. the filtered set that will actually be scored),
    avoiding a full ~800-TF scan when only ~90 matter. ChIP columns are always
    built (cheap). Other TFs still appear as columns but stay zero.

    Strategy per TF:
      - if a wild-type ChIP-seq peak file exists (OSKM): use peak counts.
      - elif genome + JASPAR motif exist: use motif-hit counts.
      - else: TF gets a zero column (reported, not silently dropped).
    Then multiply by accessibility mask s_i and activator/repressor sign a_m.
    """
    genes = list(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    activity = dict(DEFAULT_ACTIVITY)
    if activity_map:
        activity.update(activity_map)

    # ---- 1. promoters ----
    tss = load_tss(refgene_path)
    proms = promoter_intervals(tss, genes, window=window, include_body=include_body)

    # ---- 2. inventory GSE99009 BEDs ----
    beds = glob.glob(os.path.join(gse_dir, "**", "*.bed"), recursive=True) \
        + glob.glob(os.path.join(gse_dir, "**", "*.bed.gz"), recursive=True) \
        + glob.glob(os.path.join(gse_dir, "**", "*Peak*"), recursive=True)
    beds = sorted(set(beds))
    def has(tok, fn): return tok.lower() in os.path.basename(fn).lower()
    def ignored(fn): return any(has(t, fn) for t in IGNORE_TOKENS)

    chip_files = {}      # TF symbol -> bed path
    access_files = []
    for fn in beds:
        if ignored(fn):
            continue
        for tok, sym in CHIP_TF_PATTERNS.items():
            if has(tok, fn):
                chip_files.setdefault(sym, fn)
        if any(has(t, fn) for t in ACCESS_TOKENS):
            access_files.append(fn)
    if verbose:
        print(f"  GSE99009: {len(beds)} bed-like files; "
              f"ChIP TFs found: {sorted(chip_files)}; "
              f"accessibility tracks: {len(access_files)}")

    # ---- 3. accessibility mask s_i (union of active peaks) ----
    s = np.zeros(len(genes))
    if access_files:
        access_peaks = {}
        for fn in access_files:
            pk = load_bed_peaks(fn)
            for c, arr in pk.items():
                access_peaks.setdefault(c, []).append(arr)
        access_peaks = {c: np.vstack(v) for c, v in access_peaks.items()}
        access_peaks = {c: a[np.argsort(a[:, 0])] for c, a in access_peaks.items()}
        for g, prom in proms.items():
            s[gidx[g]] = 1.0 if any_overlap(prom, access_peaks) else 0.0
        print(f"  accessibility: {int(s.sum())}/{len(genes)} promoters open")
    else:
        print("  accessibility: no active-mark tracks found -> s_i = 1 for all "
              "(no masking)")
        s[:] = 1.0

    # ---- 4. which TFs we can build, and how ----
    tfs_present, modes = [], {}
    for t in tf_list:
        if t not in gidx:           # TF must be a gene (it's a column target? no:
            pass                    # TF need not be in `genes`; it's a regulator)
        if t in chip_files:
            tfs_present.append(t); modes[t] = "chip"
        elif jaspar_path and genome_path:
            tfs_present.append(t); modes[t] = "motif"
        else:
            tfs_present.append(t); modes[t] = "none"
    n_chip = sum(1 for m in modes.values() if m == "chip")
    n_motif = sum(1 for m in modes.values() if m == "motif")
    n_none = sum(1 for m in modes.values() if m == "none")
    print(f"  B columns: {n_chip} ChIP-backed, {n_motif} motif-scanned, "
          f"{n_none} zero (no data)")

    # ---- 5. ChIP columns ----
    B = np.zeros((len(genes), len(tfs_present)))
    tcol = {t: j for j, t in enumerate(tfs_present)}
    for t, fn in chip_files.items():
        if t not in tcol:
            continue
        peaks = load_bed_peaks(fn)
        peaks = {c: a[np.argsort(a[:, 0])] for c, a in peaks.items()}
        col = tcol[t]
        for g, prom in proms.items():
            B[gidx[g], col] = count_overlaps(prom, peaks)

    # ---- 6. motif columns (optional) ----
    if jaspar_path and genome_path and n_motif > 0:
        pfms = parse_jaspar(jaspar_path)
        # which motif-mode TFs do we actually need?
        motif_tfs = [t for t in tfs_present if modes[t] == "motif"]
        if restrict_tfs is not None:
            rset = set(restrict_tfs)
            motif_tfs = [t for t in motif_tfs if t in rset]
        # match to JASPAR by uppercased symbol (handles mouse->vertebrate casing)
        need = [t for t in motif_tfs if t.upper() in pfms]
        missing = [t for t in motif_tfs if t.upper() not in pfms]

        # ---- PRE-FLIGHT REPORT (before the slow scan) ----
        n_prom = sum(1 for p in proms.values() if p is not None)
        print(f"\n  === MOTIF SCAN PRE-FLIGHT ===")
        print(f"  TFs to scan (have JASPAR motif): {len(need)} / {len(motif_tfs)} "
              f"requested")
        if missing:
            show = missing[:15]
            print(f"  no JASPAR motif for {len(missing)} TFs"
                  f"{' (e.g. ' + ', '.join(show) + ')' if show else ''}")
        print(f"  promoters to scan: {n_prom}  x  window {2*window}bp  x both strands")
        approx = len(need) * n_prom
        print(f"  ~{approx:,} (TF x promoter) scans. If this is near zero, STOP "
              f"(Ctrl+C): name matching failed.")
        if len(need) == 0:
            print("  !! ZERO matchable motifs -> skipping scan (B would be all-zero "
                  "for motif TFs). Check JASPAR file / TF naming.")
        print(f"  ============================\n")

        if need:
            genome = load_genome(genome_path)
            lo_cache = {}
            for t in need:
                lo = pfm_to_logodds(pfms[t.upper()])
                lo_cache[t] = (lo, lo[::-1, ::-1])   # fwd, revcomp
            done = 0
            for g, prom in proms.items():
                if prom is None:
                    continue
                c, st, en = prom
                seq = get_seq(genome, c, st, en)
                if not seq:
                    continue
                sidx = _seq_to_idx(seq)
                for t in need:
                    lo_f, lo_r = lo_cache[t]
                    B[gidx[g], tcol[t]] = scan_motif(sidx, lo_f, lo_r, motif_threshold)
                done += 1
                if verbose and done % 200 == 0:
                    print(f"    scanned {done}/{n_prom} promoters...")

    # ---- 7. apply accessibility mask and activator/repressor sign ----
    B = B * s[:, None]
    for t in tfs_present:
        a = activity.get(t, +1)            # default activator if unknown
        B[:, tcol[t]] *= a

    nz = np.sum(np.abs(B).sum(axis=0) > 0)
    print(f"  B(binding): {B.shape}, {nz} nonzero TF columns")
    return B, tfs_present
