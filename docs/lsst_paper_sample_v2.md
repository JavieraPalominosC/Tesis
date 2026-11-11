# Sample del paper (Förster et al., in preparation)

El paper **"ALeRCE joint class-distance estimator using fast Gaussian Process estimation and causal networks"** todavía no está publicado. Lo que sigue es el texto de los README que vienen dentro de `lsst_paper_sample_v2.tar` y `lsst_paper_sample_v2_gp.tar`.

---

# ALeRCE LSST classifier: simulated training sample (v2)

Simulated LSST light curves of the objects used in the paper "ALeRCE joint class-distance estimator using fast
Gaussian Process estimation and causal networks" (Förster et al., in preparation). Simulations: ALCeS (SNANA,
LSST DDF and WFD cadences).

## Contents

- `sample.parquet` (same table as `sample.csv`): one row per object, 26,000 rows.
  26 classes x 2 surveys (DDF, WFD) x 500 objects.
- `phot/<SET>.parquet`: the simulated photometry of those objects, one file per class and survey (`<SET>` = the first
  part of `uid`, e.g. `SNIASALT2_DDF`, `RRL_WFD`). There are 52 files, each with 500 objects.

## sample table: main columns

| column | meaning |
|---|---|
| `uid` | object id: `<SET>/<run>/<SNID>` (transients) or `<SET>/w<wave>/<run>/<SNID>` (non-transients); also in `phot/` |
| `class`, `survey`, `kind` | class name; DDF or WFD; `transient` or `nontransient` |
| `split`, `fold` | `cv` (folds 0-4, 400 per class and survey) or `test` (100 per class and survey). The paper reports only the cross-validation folds; the test split was never used. Please keep it that way and do not tune on it. |
| `dist_var`, `dist`, `dist_bin` | distance variable of the paper: `log10 z` for transients and AGN, `mu + A_V` (mag, A_V = 3.1 E(B-V)_MW) for Galactic classes; its value; the class's distance bin (0-9, equal width per class) |
| `MJD_TRIGGER` | trigger time (see below) |
| everything else | simulation truth copied from the ALCeS selection files (redshift `z`/`z_true`, `ebv_mw`, host extinction, SALT2 parameters, `NT_D`, `GLAT`, `LCLIB_*`, `W_EXPECTED`, ...) |

## photometry columns

Raw SNANA output: `MJD`, `FLT` (u g r i z Y), `FLUXCAL` and `FLUXCALERR` (zero point 27.5), `FLUXTRUE` (noiseless model
flux, truth only), `MAG`, `PHOTFLAG` (detection = `PHOTFLAG & 4096`), plus `run`, `SNID` (and `wave`) and `uid`.
Every observation of the survey at the object's position is included, also those without a detection (forced photometry).

## How the paper uses the light curves

- Trigger: the first detection of the second detected epoch (epochs separated by gaps longer than 10 minutes).
- Light curve: every observation from -100 d to +300 d of the trigger (observer frame); the 1-10 yr results use
  everything after the trigger up to N years.
- Non-transients are kept only if the survey observed their position during the whole window, from 100 d before to
  300 d after the trigger (this is already applied to this sample).
- Transients: 500 detected objects per class and survey, drawn uniformly in each class's log z range. AGN, dwarf novae
  and M dwarfs are the re-simulated (long-span) versions of the paper's "v2" round.

## Caveats (see the paper)

- The DDF results of RRL, Mira, Cepheid and delta Scuti are helped by the cadence of the dense deep field.
- Cepheid, delta Scuti and dwarf novae have E(B-V) = 0 in the simulations; Mira periods (300-640 d) exceed the window.
- The classes are balanced, not in their expected LSST proportions (`W_EXPECTED` gives expected numbers for the
  non-transients).

---

# ALeRCE LSST classifier: GP fits of the simulated training sample (v2)

Gaussian-process fits of the light curves in `lsst_paper_sample_v2` (same 26,000 objects: 26 classes x DDF/WFD x
500), as used by the image network of the paper "ALeRCE joint class-distance estimator using fast Gaussian Process
estimation and causal networks" (Förster et al., in preparation). Use together with `lsst_paper_sample_v2/sample.parquet`
(classes, splits, folds, truth).

## Contents

- `gp/<SET>_mean.npy`, `gp/<SET>_var.npy`: float16 arrays of shape [500, 200, 300] per class and survey
  (`<SET>` = first part of `uid`, e.g. `SNIASALT2_DDF`). Axis 1 is ln wavelength, axis 2 is time. They are stored
  divided by a per-object scale.
- `gp_index.parquet` (same as `gp_index.csv`): one row per object, in the same order as `sample.parquet`:
  `uid`, `set`, `row` (index along axis 0 of the SET arrays), `scale_mean`, `scale_var`, and the fitted kernel
  hyperparameters `lnA`, `lnlt`, `lnll` (log amplitude, log time length scale, log ln-wavelength length scale).
- `grid.npz`: `tgrid` (300 values, days from `MJD_TRIGGER`, observer frame, -100 to +300) and `lngrid` (200 values,
  ln of wavelength in Angstrom, 3200 to 10200 A, evenly spaced in ln lambda).

## Reconstructing a fit

```python
import numpy as np, pandas as pd
ix = pd.read_parquet("gp_index.parquet")
r = ix.iloc[0]
mean = np.load(f"gp/{r.set}_mean.npy", mmap_mode="r")[r.row].astype(np.float32) * r.scale_mean   # FLUXCAL units
var  = np.load(f"gp/{r.set}_var.npy",  mmap_mode="r")[r.row].astype(np.float32) * r.scale_var    # FLUXCAL^2
g = np.load("grid.npz"); t, lam = g["tgrid"], np.exp(g["lngrid"])
```

FLUXCAL is the SNANA calibrated flux (zero point 27.5), as in the photometry files.

## How the fits were made

- 2D GP in (time, ln wavelength): Matern-3/2 kernel in time times a squared-exponential kernel in ln wavelength,
  with each LSST band at its effective wavelength. Exact O(n) solver (gpimg); hyperparameters fitted per object by
  maximum likelihood.
- Data: every observation from -100 to +300 d around the trigger (first detection of the second detected epoch),
  detections and non-detections (forced photometry).
- The grid and the window are the same for every class: nothing about the class or the distance enters the fit.
- These are the fits to the full -100..+300 d window. The paper's causal network instead refits the GP to the data
  available up to each cut time; those fits are not included and can be recomputed from the photometry.
