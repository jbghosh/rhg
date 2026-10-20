# rhg_kwant_core.py
# Worker-side model, Kwant builders and observables for notebook 05 (RHG thermal-metal study).
# The notebook writes this file to disk so that joblib workers import identical code.
import json, functools, warnings
import numpy as np
import kwant

warnings.filterwarnings("ignore", message=".*MUMPS.*")

SQ3 = np.sqrt(3.0)
TZ = np.diag([1.0, -1.0]).astype(complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)

# =============================================================================
# 1. k-space model: identical functional form to RHG_warped_finiteQ.ipynb (V17)
# =============================================================================
B = np.array([[1.0, 0.0], [-0.5, SQ3 / 2], [-0.5, -SQ3 / 2]])
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, SQ3 / 2])
GM = 2 * np.pi * np.linalg.inv(np.array([A1, A2])).T          # rows: G1, G2
AREA_PER_SITE = SQ3 / 2


def _dots(kx, ky):
    return [kx * b[0] + ky * b[1] for b in B]


def rho2(kx, ky):
    return (4.0 / 3.0) * (3.0 - sum(np.cos(x) for x in _dots(kx, ky)))


def g3(kx, ky):
    return sum(np.sin(x) for x in _dots(kx, ky))


def sxy(kx, ky):
    d = _dots(kx, ky)
    return ((2.0 / 3.0) * sum(B[i, 0] * np.sin(d[i]) for i in range(3)),
            (2.0 / 3.0) * sum(B[i, 1] * np.sin(d[i]) for i in range(3)))


def xi_normal(kx, ky, p):
    r2 = rho2(kx, ky)
    return (r2 - p["r_in"] ** 2) * (r2 - p["r_out"] ** 2) + p["w"] * g3(kx, ky)


def gap_k(kx, ky, p):
    sx, sy = sxy(kx, ky)
    c, q = p["chi"], p["q"]
    return p["D0"] * (sx + 1j * c * sy) * (sx - q + 1j * c * sy) * (sx + q + 1j * c * sy)


def bdg_fields(kx, ky, p):
    """(xi_s, xi_a, Delta) for pair momentum (Qx, Qy), following Yang and Zhang."""
    Qx, Qy = p.get("Qx", 0.0), p.get("Qy", 0.0)
    xp = xi_normal(kx + Qx / 2, ky + Qy / 2, p)
    xm = xi_normal(-kx + Qx / 2, -ky + Qy / 2, p)
    return 0.5 * (xp + xm), 0.5 * (xp - xm), gap_k(kx, ky, p)


def bloch_bdg(kx, ky, p):
    """[[xi(k+Q/2), Delta(k)], [Delta*(k), -xi(-k+Q/2)]] with shape (..., 2, 2)."""
    Qx, Qy = p.get("Qx", 0.0), p.get("Qy", 0.0)
    xp = xi_normal(kx + Qx / 2, ky + Qy / 2, p)
    xm = xi_normal(-kx + Qx / 2, -ky + Qy / 2, p)
    D = gap_k(kx, ky, p)
    H = np.zeros(np.shape(kx) + (2, 2), complex)
    H[..., 0, 0] = xp; H[..., 1, 1] = -xm; H[..., 0, 1] = D; H[..., 1, 0] = np.conj(D)
    return H


def bz_grid(Nk):
    f = np.arange(Nk) / Nk
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    return F1 * GM[0, 0] + F2 * GM[1, 0], F1 * GM[0, 1] + F2 * GM[1, 1]


def fhs_chern(xi_s, D):
    """FHS Chern number of the lower band of d=(Re D, -Im D, xi_s) (same code as V17) and the
    plaquette fluxes F[i, j] of the plaquette with lower-left corner (i, j)."""
    dx, dy, dz = np.real(D), -np.imag(D), xi_s
    n = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    uA = np.stack([dz - n, dx + 1j * dy], -1)
    uB = np.stack([-(dx - 1j * dy), dz + n], -1)
    u = np.where((dz <= 0)[..., None], uA, uB)
    nrm = np.linalg.norm(u, axis=-1, keepdims=True)
    u = u / np.where(nrm == 0, 1.0, nrm)

    def lk(a, b):
        s = np.sum(np.conj(a) * b, -1)
        return s / np.where(np.abs(s) == 0, 1.0, np.abs(s))
    U1 = lk(u, np.roll(u, -1, 0)); U2 = lk(u, np.roll(u, -1, 1))
    F = np.angle(U1 * np.roll(U2, -1, 0) / np.roll(U1, -1, 1) / U2)
    return F.sum() / (2 * np.pi), F


def kspace_summary(p, Nk_C=241, Nk_gap=481):
    kx, ky = bz_grid(Nk_C)
    xs, xa, D = bdg_fields(kx, ky, p)
    C, F = fhs_chern(xs, D)
    kx, ky = bz_grid(Nk_gap)
    xs, xa, D = bdg_fields(kx, ky, p)
    eta = np.sqrt(xs ** 2 + np.abs(D) ** 2)
    dl = eta - np.abs(xa)
    return dict(C=float(C), maxF=float(np.abs(F).max()), delta_BdG=float(dl.min()),
                eta_min=float(eta.min()), pocket_area_frac=float((dl < 0).mean()))


# ---- exact thermal kernel of V17, Eq. (kernel-G), extended as an odd function
def _li2_neg_exp(a):
    z = -np.exp(-2 * a)
    k = np.arange(1, 300)
    return np.sum(z[..., None] ** k / k ** 2, axis=-1)


def G_kernel(x):
    x = np.asarray(x, float)
    a = np.minimum(np.abs(x), 40.0)
    out = 1 - (12 / np.pi ** 2) * (a ** 2 * (1 - np.tanh(a)) + 2 * a * np.log1p(np.exp(-2 * a))
                                    - _li2_neg_exp(a))
    out = np.where(np.abs(x) >= 40.0, 1.0, out)
    return np.sign(x) * out


def kspace_thermal(p, Nk=401, thetas=(), E_grid=None):
    """Clean two-band thermal Hall of V17 on a plaquette grid.
    Returns C_band, curvature-blocking T->0 value, A_Omega, kappa/(T q_M) via the branch weight W_Q
    (V17 Eq. branch-weight), the Fermi-sea Hall function s(E), and kappa via the s(E) convolution."""
    kx, ky = bz_grid(Nk)
    xs, xa, D = bdg_fields(kx, ky, p)
    C, F = fhs_chern(xs, D)
    eta = np.sqrt(xs ** 2 + np.abs(D) ** 2)
    av = lambda a: 0.25 * (a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))
    eta_p, xa_p = av(eta), av(xa)
    Em, Ep = xa_p - eta_p, xa_p + eta_p
    blocked = np.abs(xa_p) > eta_p
    res = dict(C_band=float(C), A_Omega=float(np.abs(F).sum() / (2 * np.pi)),
               blocking_T0=float(F[~blocked].sum() / (2 * np.pi)),
               pocket_curvature=float(F[blocked].sum() / (2 * np.pi)),
               delta_BdG=float((eta - np.abs(xa)).min()))
    kap = []
    for th in thetas:
        W = 0.5 * (G_kernel((eta_p + xa_p) / (2 * th)) + G_kernel((eta_p - xa_p) / (2 * th)))
        kap.append(float((F * W).sum() / (2 * np.pi)))
    res["kappa_WQ"] = np.array(kap)
    if E_grid is not None:
        Ff, Emf, Epf = F.ravel(), Em.ravel(), Ep.ravel()
        om = np.argsort(Emf); op = np.argsort(Epf)
        cm = np.concatenate([[0], np.cumsum(Ff[om])]); cp = np.concatenate([[0], np.cumsum(Ff[op])])
        im = np.searchsorted(Emf[om], E_grid); ip = np.searchsorted(Epf[op], E_grid)
        s = (cm[im] - cp[ip]) / (2 * np.pi)       # lower band filled minus upper band (Omega_+ = -Omega_-)
        res["s_E"] = s
        kconv = []
        for th in thetas:
            x = np.clip(E_grid / (2 * th), -300, 300)
            mdf = 1.0 / (4 * th * np.cosh(x) ** 2)
            kconv.append(float(3 / (np.pi ** 2 * th ** 2) * np.trapezoid(E_grid ** 2 * mdf * s, E_grid)))
        res["kappa_conv"] = np.array(kconv)
    return res


def normal_dos_at_EF(p, Nk=601, width=0.01):
    """Normal-state DOS per site per unit energy at E=0 (Lorentzian-free histogram estimate)."""
    kx, ky = bz_grid(Nk)
    xi = xi_normal(kx, ky, p)
    return float((np.abs(xi) < width).mean() / (2 * width))


# =============================================================================
# 2. Real-space hoppings by exact Fourier inversion of the Bloch functions
#    convention f(k) = sum_d f_d exp(-i k.d),  d = n1 A1 + n2 A2
# =============================================================================
def fourier_coeffs(func, R=4, N=24, tol=1e-12):
    kx, ky = bz_grid(N)
    vals = func(kx, ky)
    out = {}
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            d = n1 * A1 + n2 * A2
            c = np.mean(vals * np.exp(1j * (kx * d[0] + ky * d[1])))
            if abs(c) > tol:
                out[(n1, n2)] = complex(c)
    return out


def model_key(p):
    return json.dumps({k: float(p[k]) for k in ("r_in", "r_out", "w", "q", "D0", "chi")}, sort_keys=True)


@functools.lru_cache(maxsize=64)
def model_hoppings(key):
    p = json.loads(key)
    return (fourier_coeffs(lambda kx, ky: xi_normal(kx, ky, p)),
            fourier_coeffs(lambda kx, ky: gap_k(kx, ky, p)))


def reconstruct_error(p, n_test=200, seed=0):
    t, dl = model_hoppings(model_key(p))
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n_test):
        kx, ky = rng.uniform(-np.pi, np.pi, 2)
        ph = lambda n1, n2: np.exp(-1j * (kx * (n1 + n2 / 2) + ky * n2 * SQ3 / 2))
        e = sum(c * ph(*d) for d, c in t.items())
        dd = sum(c * ph(*d) for d, c in dl.items())
        err = max(err, abs(e - xi_normal(kx, ky, p)), abs(dd - gap_k(kx, ky, p)))
    rng_of = lambda dct: max(max(abs(a), abs(b), abs(a + b)) for (a, b) in dct)
    odd = max(abs(dl[k] + dl.get((-k[0], -k[1]), 0)) for k in dl)
    return dict(max_err=float(err), normal_range=rng_of(t), pairing_range=rng_of(dl),
                n_normal=len(t), n_pairing=len(dl), pairing_oddness=float(odd))


# =============================================================================
# 3. Lattice geometry. Integer coordinates (X, j): x = X/2, y = j*sqrt(3)/2, X = j (mod 2).
#    Kwant: rectangular cell (1, sqrt3) with two sublattices.
# =============================================================================
LAT = kwant.lattice.general([(1, 0), (0, SQ3)], [(0, 0), (0.5, SQ3 / 2)], norbs=2, name=["a", "b"])
SUBL = LAT.sublattices


def site_of(X, j):
    s = j % 2
    return SUBL[s]((X - s) // 2, j // 2)


def xj_of(site):
    s = 0 if site.family is SUBL[0] else 1
    ix, iy = site.tag
    return 2 * ix + s, 2 * iy + s


def nambu_block(t, d):
    """Kwant block H[target, source] for electron hopping t and odd pairing amplitude d."""
    return np.array([[t, d], [-np.conj(d), -np.conj(t)]], dtype=complex)


def _half(n1, n2):
    return n2 > 0 or (n2 == 0 and n1 > 0)


@functools.lru_cache(maxsize=8)
def _noise(seed, nX, nJ):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(nX, nJ))


@functools.lru_cache(maxsize=32)
def domain_mask(seed, nX, nJ, ell, frac, periodic_j=True):
    """True = domain B (time-reversed partner). Smoothed Gaussian random field with correlation
    length `ell` (lattice units), thresholded so that a fraction `frac` of the grid is B."""
    from scipy import ndimage
    f = np.random.default_rng(10_000 + seed).normal(size=(nX, nJ))
    f = ndimage.gaussian_filter(f, sigma=(2 * ell, 2 * ell / SQ3),
                                mode=("nearest", "wrap" if periodic_j else "nearest"))
    return f > np.quantile(f, 1 - frac)


def _twist(M, sign):
    """Seam hopping with a twisted boundary condition psi -> exp(i phi) psi applied to the full Nambu
    spinor. This is a fictitious U(1) twist (not a charge flux): it treats H_BdG as a class-A matrix, whose
    Chern number is the class-D invariant C_BdG, as in the Bott index."""
    def val(s1, s2, phi):
        return np.exp(-1j * sign * phi) * M
    return val


def build_system(g):
    """g: geometry + model dict.
       kind: 'torus' (periodic x,y; gauge frame) | 'cylinder' (periodic y, leads along x) |
             'strip' (open y, leads along x) | 'flake' (open, no leads)
       Lx, Wy: cells (width Lx, height sqrt3*Wy, 2*Lx*Wy sites); frame: 'gauge' | 'lab'
       model: r_in r_out w q D0 chi Qx ; buffer: normal (Delta=0) cells next to each lead
       texture: None | 'wall2' (two walls, cylinder) | 'wall1' (one wall, strip) | 'random'
       wall_kind: 'sharp' | 'suppressed' | 'normal'; wall_width (lattice units); wall_lambda
       ell_d, frac_d, tex_seed for random textures.
    Kwant params at solve time: U (uniform on-site disorder amplitude), seed, mu_shift, phi (flux twist)."""
    Lx, Wy = int(g["Lx"]), int(g["Wy"])
    nX, nJ = 2 * Lx, 2 * Wy
    kind = g["kind"]
    per_x = kind == "torus"
    per_j = kind in ("torus", "cylinder")
    gauge = g.get("frame", "lab") == "gauge"
    Qx = float(g.get("Qx", 0.0))
    t, dl = model_hoppings(model_key(g))
    buf = int(g.get("buffer", 0)) if kind in ("cylinder", "strip") else 0
    x_sc0, x_sc1 = buf, Lx - buf
    in_sc = lambda X: x_sc0 <= X / 2 < x_sc1

    tex = g.get("texture")
    H_y = nJ * SQ3 / 2
    if tex == "wall2":
        walls = [H_y / 4, 3 * H_y / 4]
        maskB = np.broadcast_to(((np.arange(nJ) * SQ3 / 2 >= walls[0]) & (np.arange(nJ) * SQ3 / 2 < walls[1]))[None, :], (nX, nJ))
    elif tex == "wall1":
        walls = [H_y / 2]
        maskB = np.broadcast_to((np.arange(nJ) * SQ3 / 2 < walls[0])[None, :], (nX, nJ))
    elif tex == "random":
        walls = []
        maskB = domain_mask(int(g["tex_seed"]), nX, nJ, float(g["ell_d"]), float(g["frac_d"]), per_j)
    else:
        walls, maskB = [], None
    wk, ww, wl = g.get("wall_kind", "sharp"), float(g.get("wall_width", 0.0)), float(g.get("wall_lambda", 1.0))

    def wall_dist(X, j):
        if not walls:
            return np.inf
        y = j * SQ3 / 2
        if per_j:
            return min(min(abs(y - w), H_y - abs(y - w)) for w in walls)
        return min(abs(y - w) for w in walls)

    def pair_scale(X, j):
        if wk in ("suppressed", "normal") and ww > 0 and wall_dist(X, j) < ww / 2:
            return 0.0 if wk == "normal" else wl
        return 1.0

    def inB(X, j):
        return maskB is not None and in_sc(X) and bool(maskB[X % nX, j % nJ])

    sites = {}
    for j in range(nJ):
        s = j % 2
        for ix in range(Lx):
            sites[(2 * ix + s, j)] = site_of(2 * ix + s, j)

    syst = kwant.Builder()
    t0 = t.get((0, 0), 0.0).real

    def onsite(site, U, seed, mu_shift):
        X, j = xj_of(site)
        v = 0.0 if U == 0 else U * _noise(int(seed), nX, nJ)[X % nX, j % nJ]
        return (t0 - mu_shift + v) * TZ

    for st in sites.values():
        syst[st] = onsite

    for (n1, n2) in set(list(t) + list(dl)):
        if not _half(n1, n2):
            continue
        dX, dj = 2 * n1 + n2, n2
        dxc = n1 + n2 / 2
        tc, dc = t.get((n1, n2), 0.0), dl.get((n1, n2), 0.0)
        te0 = tc * (np.exp(-1j * Qx * dxc / 2) if gauge else 1.0)
        for (X, j), st in sites.items():
            X2, j2 = X + dX, j + dj
            crossed = 0
            if per_x:
                X2 %= nX
            if per_j and j2 >= nJ:
                j2 -= nJ; crossed = 1
            tgt = sites.get((X2, j2))
            if tgt is None:
                continue
            te = te0
            de = 0.0
            if dc != 0 and (buf == 0 or (in_sc(X) and in_sc(X + dX))):
                sc = 0.5 * (pair_scale(X, j) + pair_scale(X + dX, j + dj))
                de = dc * sc * (1.0 if gauge else np.exp(1j * Qx * (X / 2 + dxc / 2)))
            if maskB is not None:
                bA, bB = inB(X, j), inB(X + dX, j + dj)
                if bA and bB:
                    te, de = np.conj(te), np.conj(de)
                elif bA != bB:
                    te, de = np.real(te), np.real(de)
            if te == 0 and de == 0:
                continue
            M = nambu_block(te, de)
            syst[tgt, st] = _twist(M, +1) if (crossed and kind == "cylinder") else M

    if kind in ("cylinder", "strip"):
        P = 2
        lead_type = g.get("lead_type", "chains")
        tL = float(g.get("t_lead", 1.0))
        for direction in (-1, +1):
            lead = kwant.Builder(kwant.TranslationalSymmetry((direction * P, 0)))
            lsites = {}
            for j in range(nJ):
                s = j % 2
                for ix in range(P):
                    lsites[(2 * ix + s, j)] = site_of(2 * ix + s, j)
            if lead_type == "chains":
                # decoupled 1D chains along x (one per row): mode content independent of the twist phi,
                # as required by the scattering formula for the Chern number (Fulga et al. 2012)
                for st in lsites.values():
                    lead[st] = 0.0 * TZ
                for (X, j), st in lsites.items():
                    lead[site_of(X + 2, j), st] = nambu_block(-tL, 0.0)
            else:
                # normal metal with the same xi(k) and Delta = 0
                for st in lsites.values():
                    lead[st] = t0 * TZ
                for (n1, n2), tc in t.items():
                    if not _half(n1, n2):
                        continue
                    dX, dj = 2 * n1 + n2, n2
                    for (X, j), st in lsites.items():
                        j2, crossed = j + dj, 0
                        if per_j and j2 >= nJ:
                            j2 -= nJ; crossed = 1
                        elif not (0 <= j2 < nJ):
                            continue
                        M = nambu_block(tc, 0.0)
                        lead[site_of(X + dX, j2), st] = _twist(M, +1) if crossed else M
            syst.attach_lead(lead)
    return syst.finalized()


# ---- calibration model: square-lattice spinless p+ip (V17 Methods anchor), random chemical potential
def build_pip(g):
    """Square lattice, d = (D sin kx, D sin ky, 2t(2 - cos kx - cos ky) - mu). C = +1 (0 < mu < 4t) and
    -1 (4t < mu < 8t) in the V17 FHS convention. Cylinder periodic in y, normal leads (Delta = 0, mu_lead)."""
    L, W = int(g["Lx"]), int(g["Wy"])
    t, D, mu_bar, mu_lead = g.get("t", 1.0), g["D"], g["mu_bar"], g.get("mu_lead", g["mu_bar"])
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

    def onsite(site, U, seed, mu_shift):
        x, y = site.tag
        v = 0.0 if U == 0 else U * _noise(int(seed), L, W)[x, y]
        return (4 * t - mu_bar - v - mu_shift) * TZ
    for x in range(L):
        for y in range(W):
            syst[lat(x, y)] = onsite
    hx = -t * TZ + 0.5j * D * SX
    hy = -t * TZ + 0.5j * D * SY
    for x in range(L):
        for y in range(W):
            if x + 1 < L:
                syst[lat(x + 1, y), lat(x, y)] = hx
            if y + 1 < W:
                syst[lat(x, y + 1), lat(x, y)] = hy
            else:
                syst[lat(x, 0), lat(x, y)] = _twist(hy, +1)
    for direction in (-1, +1):
        # decoupled 1D chains (see build_system)
        lead = kwant.Builder(kwant.TranslationalSymmetry((direction, 0)))
        for y in range(W):
            lead[lat(0, y)] = 0.0 * TZ
        for y in range(W):
            lead[lat(1, y), lat(0, y)] = -t * TZ
        syst.attach_lead(lead)
    return syst.finalized()


# =============================================================================
# 4. Observables (worker tasks)
# =============================================================================
def gkey(g):
    return json.dumps(g, sort_keys=True)


def build_pip_flake(g):
    """Open square flake of the p+ip anchor (no leads), for the KPM Hall calibration."""
    L, t, D, mu = int(g["Lx"]), g.get("t", 1.0), g["D"], g["mu_bar"]
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

    def onsite(site, U, seed, mu_shift):
        x, y = site.tag
        v = 0.0 if U == 0 else U * _noise(int(seed), L, L)[x, y]
        return (4 * t - mu - v - mu_shift) * TZ
    for x in range(L):
        for y in range(L):
            syst[lat(x, y)] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat)] = -t * TZ + 0.5j * D * SX
    syst[kwant.builder.HoppingKind((0, 1), lat)] = -t * TZ + 0.5j * D * SY
    return syst.finalized()


@functools.lru_cache(maxsize=2)
def cached_system(key):
    g = json.loads(key)
    if g["kind"] == "pip_cylinder":
        return build_pip(g)
    if g["kind"] == "pip_flake":
        return build_pip_flake(g)
    return build_system(g)


def _params(U, seed, phi=0.0, mu_shift=0.0):
    return dict(U=float(U), seed=int(seed), mu_shift=float(mu_shift), phi=float(phi))


def det_r_winding(fsyst, E, U, seed, phis):
    """Scattering Chern number (Fulga, Hassler, Akhmerov, PRB 85, 165409 (2012)): winding of det r(phi)
    of the left-lead reflection matrix as the flux phi threading the cylinder runs over 2 pi."""
    dets = []
    for ph in phis:
        S = kwant.smatrix(fsyst, E, params=_params(U, seed, ph))
        r = S.submatrix(0, 0)
        sgn, logabs = np.linalg.slogdet(r)
        dets.append(sgn * np.exp(logabs))
    dets = np.array(dets)
    ang = np.unwrap(np.angle(np.append(dets, dets[0])))
    return float((ang[-1] - ang[0]) / (2 * np.pi)), float(np.abs(dets).min())


def task_transport(key, seeds, U, E_list, phis=None, E_wind=1e-4):
    """Two-terminal transmission T_{R<-L}(E) (Nambu trace = Majorana units) for each seed, and,
    if `phis` is given, the scattering winding number at E_wind."""
    fsyst = cached_system(key)
    out = []
    for sd in seeds:
        rec = dict(seed=int(sd), U=float(U))
        T = []
        for E in E_list:
            try:
                S = kwant.smatrix(fsyst, float(E), params=_params(U, sd))
                T.append(S.transmission(1, 0))
            except Exception:
                T.append(np.nan)
        rec["T"] = np.array(T)
        if phis is not None:
            try:
                rec["Q"], rec["min_abs_det_r"] = det_r_winding(fsyst, E_wind, U, sd, phis)
            except Exception:
                rec["Q"], rec["min_abs_det_r"] = np.nan, np.nan
        out.append(rec)
    return out


def task_winding(key, seeds, U, phis, E=1e-4):
    """Scattering Chern number Q and min_phi |det r| for each seed (system must have chain leads)."""
    fsyst = cached_system(key)
    out = []
    for sd in seeds:
        try:
            Q, md = det_r_winding(fsyst, E, U, sd, phis)
        except Exception:
            Q, md = np.nan, np.nan
        out.append(dict(seed=int(sd), U=float(U), Q=Q, min_abs_det_r=md))
    return out


def task_bott(key, seeds, U):
    """Bott index of all negative-energy BdG states on a torus, plus low-energy level counts."""
    fsyst = cached_system(key)
    g = json.loads(key)
    pos = np.array([s.pos for s in fsyst.sites])
    ux = np.repeat(np.exp(2j * np.pi * pos[:, 0] / g["Lx"]), 2)
    uy = np.repeat(np.exp(2j * np.pi * pos[:, 1] / (SQ3 * g["Wy"])), 2)
    out = []
    for sd in seeds:
        H = fsyst.hamiltonian_submatrix(params=_params(U, sd))
        w, v = np.linalg.eigh(H)
        occ = v[:, w < 0]
        Ux = occ.conj().T @ (ux[:, None] * occ)
        Uy = occ.conj().T @ (uy[:, None] * occ)
        ev = np.linalg.eigvals(Uy @ Ux @ Uy.conj().T @ Ux.conj().T)
        absw = np.abs(w)
        out.append(dict(seed=int(sd), U=float(U), bott=float(np.sum(np.angle(ev)) / (2 * np.pi)),
                        min_abs_E=float(absw.min()), n_sites=len(fsyst.sites),
                        n_lt=[int((absw < e).sum()) for e in (0.005, 0.01, 0.02, 0.04)]))
    return out


def task_kpm_dos(key, seed, U, energies, num_moments, num_vectors=4):
    """KPM density of states per site per unit energy (Nambu: counts both particle and hole)."""
    fsyst = cached_system(key)
    sd = kwant.kpm.SpectralDensity(fsyst, params=_params(U, seed), num_moments=int(num_moments),
                                   num_vectors=int(num_vectors), rng=int(seed))
    return np.real(sd(np.asarray(energies))) / len(fsyst.sites)


def task_kpm_sxy(key, U, seed, energies, num_moments, num_vectors, temperature, frac=0.5):
    """Real-space Kubo-Bastin Hall function s(E) of the BdG Hamiltonian on an open flake, sampled in
    the central region (fraction `frac` of each side). Raw Kwant output divided by the sampled area.
    The sign/normalization is calibrated in the notebook on the p+ip anchor."""
    fsyst = cached_system(key)
    g = json.loads(key)
    pip = g["kind"] == "pip_flake"
    Lx, Hy = (g["Lx"], g["Lx"]) if pip else (g["Lx"], SQ3 * g["Wy"])
    aps = 1.0 if pip else AREA_PER_SITE
    lo, hi = (1 - frac) / 2, (1 + frac) / 2
    where = lambda s: (lo * Lx <= s.pos[0] < hi * Lx) and (lo * Hy <= s.pos[1] < hi * Hy)
    n_where = sum(1 for s in fsyst.sites if where(s))
    vf = kwant.kpm.RandomVectors(fsyst, where=where, rng=int(seed))
    cond = kwant.kpm.conductivity(fsyst, alpha="x", beta="y", params=_params(U, seed),
                                  num_moments=int(num_moments), num_vectors=int(num_vectors),
                                  vector_factory=vf, rng=int(seed))
    vals = np.array([cond(mu=float(e), temperature=float(temperature)) for e in energies])
    return np.real(vals) / (n_where * aps)


def current_map(key, E, U=0.0, seed=0, lead=0):
    """Quasiparticle probability current of all scattering states injected from `lead` at energy E.
    Returns hopping midpoints, direction vectors and current values."""
    fsyst = cached_system(key)
    wf = kwant.wave_function(fsyst, energy=E, params=_params(U, seed))
    psi = wf(lead)
    J = kwant.operator.Current(fsyst)
    cur = sum(J(p, params=_params(U, seed)) for p in psi)
    pos = np.array([s.pos for s in fsyst.sites])
    hops = np.array(fsyst.graph.tail_head_pairs) if hasattr(fsyst.graph, "tail_head_pairs") else \
        np.array([(fsyst.graph.tail(i), fsyst.graph.head(i)) for i in range(fsyst.graph.num_edges)])
    return pos, hops, np.asarray(cur)


def thermal_from_T(E, T, theta):
    """K/(q_M T) = (6/(pi^2 theta^2)) int_0^inf dE E^2 (-df/dE) T(E), Nambu-trace transmission T(E)=T(-E)."""
    x = np.clip(E / (2 * theta), -300, 300)
    mdf = 1.0 / (4 * theta * np.cosh(x) ** 2)
    return float(6 / (np.pi ** 2 * theta ** 2) * np.trapezoid(E ** 2 * mdf * T, E))


def cut_current_profile(key, E, U=0.0, seed=0, lead=0):
    """Quasiparticle probability current through the vertical cut x = Lx/2 for all scattering states
    injected from `lead` at energy E. Returns (y_mid, I) per crossing hopping; sum(I) = T_{other<-lead}."""
    fsyst = cached_system(key)
    prm = _params(U, seed)
    psi = kwant.wave_function(fsyst, energy=E, params=prm)(lead)
    g = json.loads(key)
    xc = g["Lx"] / 2 - 0.25
    pos = np.array([s.pos for s in fsyst.sites])
    out = []
    for i in range(len(fsyst.sites)):
        if pos[i, 0] >= xc:
            continue
        for jn in fsyst.graph.out_neighbors(i):
            if pos[jn, 0] < xc:
                continue
            Hji = np.asarray(fsyst.hamiltonian(jn, i, params=prm))
            pj, pi = psi[:, 2 * jn:2 * jn + 2], psi[:, 2 * i:2 * i + 2]
            Iij = 2 * np.imag(np.einsum("ma,ab,mb->", pj.conj(), Hji, pi))
            out.append((0.5 * (pos[i, 1] + pos[jn, 1]), Iij))
    return np.array(out)


def dispatch(name, kw):
    return globals()[name](**kw)
