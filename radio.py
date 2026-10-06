"""Brainwave Radio: synthetic EEG that drifts from relaxed -> focused -> deep sleep.

Run:  python radio.py   ->  assets/brainwave_radio.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram

BG, PANEL, INK, MUTE = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
PINK, LAV, MINT, PEACH, SKY = "#ff7eb6", "#b794f6", "#7ee8c7", "#ffb86b", "#79c0ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE,
    "ytick.color": MUTE, "axes.edgecolor": "#30363d", "font.size": 10,
})

FS, DUR = 256, 60
rng = np.random.default_rng(42)
t = np.arange(DUR * FS) / FS

BANDS = {  # name: (lo Hz, hi Hz, amplitude in [relaxed, focused, asleep])
    "delta": (1, 4, [0.3, 0.15, 2.0]),
    "theta": (4, 8, [0.8, 0.3, 0.5]),
    "alpha": (8, 12, [1.8, 0.4, 0.1]),
    "beta": (13, 30, [0.3, 1.2, 0.05]),
    "gamma": (30, 45, [0.1, 0.7, 0.02]),
}
STATES = ["relaxed", "focused", "deep sleep"]
BAND_COLORS = {"delta": LAV, "theta": SKY, "alpha": MINT, "beta": PEACH, "gamma": PINK}


def pink_noise(n):
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def band_signal(lo, hi, levels, n_osc=6):
    steps = np.repeat(levels, DUR * FS // len(levels)).astype(float)
    amp = gaussian_filter1d(steps, 1.5 * FS)  # smooth the handovers between states
    sig = np.zeros_like(t)
    for _ in range(n_osc):
        f = rng.uniform(lo, hi)
        phase = rng.uniform(0, 2 * np.pi)
        env = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
        sig += env * np.sin(2 * np.pi * f * t + phase)
    return amp * sig / np.sqrt(n_osc)


def main():
    os.makedirs("assets", exist_ok=True)
    eeg = 0.2 * pink_noise(t.size)
    for lo, hi, levels in BANDS.values():
        eeg += band_signal(lo, hi, levels)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6], hspace=0.35, wspace=0.18)

    for col, (name, centre) in enumerate(zip(STATES, (10, 30, 50))):
        ax = fig.add_subplot(gs[0, col])
        i0 = int(centre * FS)
        seg = eeg[i0:i0 + 3 * FS]
        tt = np.arange(seg.size) / FS
        color = (MINT, PEACH, LAV)[col]
        for lw, a in ((5, 0.08), (2.5, 0.2), (1, 1.0)):
            ax.plot(tt, seg, color=color, lw=lw, alpha=a)
        ax.set_title(name, color=color, fontweight="bold", loc="left")
        ax.set_xlabel("seconds")
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    f, tt, S = spectrogram(eeg, FS, nperseg=FS * 2, noverlap=int(FS * 1.75))
    keep = f <= 45
    db = 10 * np.log10(S[keep] + 1e-9)
    ax.pcolormesh(tt, f[keep], db, cmap="magma", shading="auto",
                  vmin=np.percentile(db, 25), vmax=np.percentile(db, 99.7))
    for name, (lo, hi, _) in BANDS.items():
        ax.axhline(hi, color="white", lw=0.4, alpha=0.3)
        ax.text(60.8, (lo + hi) / 2, name, color=BAND_COLORS[name], va="center", fontsize=10, fontweight="bold", clip_on=False)
    for x in (20, 40):
        ax.axvline(x, color="white", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram: watch the power move up and down the bands", loc="left", color=INK)

    fig.savefig("assets/brainwave_radio.png", dpi=130, bbox_inches="tight")
    print("saved assets/brainwave_radio.png")


if __name__ == "__main__":
    main()
