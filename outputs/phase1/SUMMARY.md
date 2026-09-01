# Phase 1 — Real benchmark migration

**Status: complete for `pushpins`. Two negative results, one of which retracts a
conclusion the project previously reported.**

## What changed

- **Real MVTec LOCO AD is installed and verified.** 3,651 images, all five
  categories, ground truth confirmed well-formed.
- `scripts/verify_dataset.py` — strict structural verification that fails loudly
  rather than warning politely.
- `scripts/prepare_loco.py` — pre-resize cache; loading went **84 ms → 7.2 ms
  per image (11.7×)**, which is what made multi-seed studies feasible at all.
- **Multi-seed support** (`--seeds 0,1,2`) throughout `run_ablations.py`, with
  `mean ± std` as the reporting format and single-seed values explicitly
  labelled as such.
- 254 unit tests, 16 of them on the union-of-region-masks path alone.

## The union-of-region-masks risk was real

The phase plan flagged this as the most likely silent bug. Measured across the
real dataset:

| regions per image | images |
|---:|---:|
| 1 | 906 |
| 2 | 65 |
| 3 | 10 |
| **15** | 12 |

**8.8% of annotated images carry multiple regions, and 50 of 91 `pushpins`
logical anomalies do — up to 15 each.** A loader reading only the first mask
file would lose over half that category's annotation, with no error raised and
no visibly wrong output: pixel metrics would simply be quietly too low.

Also worth recording: **LOCO masks are not `{0, 255}`.** They carry values in the
234–255 range. Any code assuming binary masks is wrong about this data; the
loader thresholds at `> 0`, and the resize cache uses NEAREST, which was verified
to introduce zero new values.

## Result 1 — the Study 1 masking ordering did NOT reproduce

Real LOCO `pushpins`, 80 epochs, **3 seeds**, `mean ± std`:

| arm | image AUROC | logical | structural |
|---|---:|---:|---:|
| `random_patch` | 0.632 ± 0.007 | 0.449 ± 0.003 | 0.837 ± 0.012 |
| `small` | 0.638 ± 0.003 | 0.448 ± 0.001 | 0.851 ± 0.005 |
| `large` | 0.638 ± 0.004 | 0.454 ± 0.004 | 0.845 ± 0.006 |
| `multiscale` | 0.641 ± 0.005 | 0.451 ± 0.003 | 0.855 ± 0.008 |
| `multiscale` + curriculum | **0.645 ± 0.007** | 0.453 ± 0.005 | **0.860 ± 0.010** |

**All five arms fall within 0.013 image AUROC of each other**, against a seed
standard deviation of ~0.005. The runner's own significance check calls the top
two `indistinguishable (0.6σ of seed noise)`.

Compare the synthetic result this project previously reported:

| | synthetic spread | real spread |
|---|---:|---:|
| best − worst image AUROC | **0.193** (0.955 vs 0.762) | **0.013** (0.645 vs 0.632) |

### What this retracts

The README previously reported, from single-seed synthetic data, that
"multi-scale masking was the weakest arm" and that "finer masking wins". **That
conclusion does not survive contact with the real benchmark.** On real data the
masking strategy has no measurable effect, and if anything the ordering is mildly
*reversed* — `multiscale + curriculum` leads and `random_patch` trails — though
every gap sits inside the noise.

So the honest statement is not "multi-scale masking failed" but **"the masking
strategy does not matter here, and the earlier ordering was an artifact of
single-seed synthetic data."** This is precisely the failure mode multi-seed
reporting was added to prevent, and it justifies that work retroactively.

It also means Phase 2a's `per_block` loss reduction — built to explain why
multi-scale lost — is answering a question that turned out not to exist on real
data. The implementation and its Study 5 remain, because the *mechanism* (block
area dominating the gradient) is real and worth measuring; but it is no longer
motivated by a deficit that needs explaining.

## Result 2 — logical anomalies score BELOW chance, and it is systematic

The far more important finding, visible in the table above:

**Logical AUROC is 0.448–0.454 across every arm.** Below chance, with seed
standard deviations of 0.001–0.005 over 15 training runs. Anomalous images are
ranked as *more normal* than normal ones, reproducibly.

Structural AUROC over the same runs is 0.837–0.860 — solidly good. So the model
is working; the *score* is inverted for one anomaly family.

That logical AUROC pins to ~0.45 regardless of masking strategy says the cause is
in **scoring, not training**.

### The mechanism

A missing pushpin leaves an empty compartment. **An empty compartment is easier
to predict than the object that belongs there.** The anomaly score counts only
"harder to predict than normal", so a missing object registers as *extra*-normal
and is pushed down the ranking.

The synthetic `screw_board` never exposed this: its logical anomalies were large
and mostly *added* structure (an extra screw, a misplaced screw), which raises
prediction error in the expected direction.

### The fix, and its status

`anomaly.deviation: {signed, absolute}` — `absolute` scores `|z|`, so a region
that is unexpectedly *easy* counts as surprising too. Added as **Study 7**, with
`tests/test_deviation.py` reproducing the failure on synthetic grids, confirming
the fix, and checking the extra-object case does not regress.

The default remains `signed`, because that is what every previously reported
number was produced under. The real-data comparison is the decisive experiment
and is reported separately.

## Caveats

- **One category.** `pushpins` was chosen because it carries the most
  multi-region logical annotation, but the other four are not yet run. The
  below-chance result may be specific to "missing small object" anomalies;
  `breakfast_box` and `juice_bottle` contain different logical failure types.
- 80 epochs per arm, matching the synthetic ablation schedule for comparability,
  not the 150-epoch headline schedule.
- Trained at 256 px from the pre-resize cache.

## Next experiments

1. **Settle Study 7 on real data** — does `absolute` lift logical AUROC above
   0.5? This is the highest-value single measurement remaining.
2. **Run the remaining four categories**, 3 seeds each, to establish whether the
   below-chance logical result is universal or `pushpins`-specific.
3. **Per-defect-type breakdown** within `logical_anomalies`. LOCO labels the
   specific defect; splitting "missing" from "added" would test the mechanism
   directly rather than by proxy.

## Reproduce

```bash
py -3.12 scripts/verify_dataset.py --root data/mvtec_loco_real --strict-loco
py -3.12 scripts/prepare_loco.py --src data/mvtec_loco_real --dst data/mvtec_loco_256
py -3.12 run_ablations.py --study masking --seeds 0,1,2 \
  --set experiment.name=real_study1 data.root=data/mvtec_loco_256 data.category=pushpins
```
