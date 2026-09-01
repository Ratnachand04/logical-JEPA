# Phase 3a — Slot bottleneck isolation gate

**Status: gate PASSED numerically, with a qualification that constrains Phase 3c.**

## What changed

- `models/slot_bottleneck.py` — slot attention implemented from scratch (slots as
  queries, patch tokens as keys/values, softmax normalised **over slots** so tokens
  are apportioned competitively). Plus a broadcast decoder so the module can be
  trained and validated standalone.
- `train_slots.py` — the isolation gate: trains `patches → slots → reconstructed
  patches` on frozen target-encoder tokens, with no predictor and no anomaly
  objective attached.
- `visualize.py --mode slots` — renders per-slot attention maps and the hard slot
  assignment over real images.

Tokens come from the frozen target encoder of a trained checkpoint, so the
bottleneck is validated on exactly the representation it would have to compress
in the full model rather than on raw pixels.

## Measured — slot-count sweep

Synthetic `screw_board`, 200 normal training images, 150 epochs, single seed.

| num_slots | entropy ratio ↓ | recon R² ↑ | dead slots | gate |
|---:|---:|---:|---:|:--|
| 4  | 0.292 | 0.944 | 0/4  | **PASS** |
| 6  | **0.190** | 0.941 | 0/6  | **PASS** |
| 8  | 0.212 | 0.940 | 0/8  | **PASS** |
| 12 | 0.959 | 0.943 | 0/12 | FAIL — attention near-uniform |

`entropy ratio` = mean entropy of each token's distribution over slots, divided by
`log(num_slots)`. 1.0 means every token is spread evenly across all slots, i.e. the
bottleneck has become an averaging layer.

**Selected: `num_slots = 6`** (lowest entropy). Reconstruction R² saturates at
~0.94 for every workable count and cannot discriminate between them, so selection
is made on how *decisively* tokens are assigned — which is the property a
cardinality question depends on.

### Two things worth recording

**Under-training reads as collapse.** At 12 epochs, `num_slots=8` scored entropy
ratio 0.981 and failed the gate. The same configuration reaches 0.212 and passes
at 150 epochs. A slot bottleneck that has not converged is indistinguishable from
one that has collapsed, so the gate must never be run on a short schedule — that
would have produced a false "stop, this does not work" conclusion.

**12 slots genuinely fails.** With more slots than the scene has components,
the competition has no incentive to specialise and attention flattens. This is
the "too many splits a component" failure mode, and it is visible in the metric
rather than only in the maps. The scene has 6 components (4 screws + product body
+ indicator strip), and the passing range 4–8 brackets that.

## The qualification — grouping is by *type*, not by *instance*

The rendered maps (`outputs/.../figures/slot_attention.png`) show the bottleneck
segmenting cleanly, but **not the way the phase plan assumed**:

- one slot takes the background,
- one takes the product body,
- **all four screws land in a single slot**, and the indicator segments share
  another.

So slot attention here performs *semantic* grouping (all objects of a kind → one
slot), not *instance* separation (each screw → its own slot). The segmentation is
sensible — it is not noise, and it is not collapsed — but the units it produces
are component **types**, not component **instances**.

### Why this matters, and what it forces

The module docstring already committed to option (ii), aggregate-only cardinality,
on the grounds that slot identity is unstable across images. That decision now has
a second, stronger justification: **per-instance counting is not merely unstable
here, it is unavailable**, because instances of the same component are not
separated in the first place.

Concretely for Phase 3c:

- a cardinality head must predict a **scalar total per region**, never a
  per-component-type breakdown — as already specified;
- the signal it can actually use is the *attention mass* of a type-slot, which
  shrinks when one of four screws goes missing, rather than a count of occupied
  slots;
- that signal is confounded with background mass, so the head should be given the
  normalised per-slot usage vector rather than raw masses.

This is a real constraint discovered by the gate, which is what the gate was for.

## Verdict against the phase plan

The plan says to stop if slots "do not segment sensibly". They do segment sensibly
— cleanly, with no dead slots and high reconstruction fidelity — so Phase 3b is
unblocked. What has changed is the *specification* of 3c, tightened by evidence
rather than assumption.

## Next experiment

1. **3b hierarchical loss** — patch term + slot term, each averaged over its own
   unit, logged separately, `lambda` ablated.
2. **3c cardinality head** — scalar-per-region, stop-gradient into the slots, fed
   the normalised slot-usage vector rather than raw attention mass.
3. Re-run this gate on real MVTec LOCO once the dataset is in place. `screw_board`
   has visually identical screws, which is close to the worst case for instance
   separation; LOCO's `breakfast_box` and `splicing_connectors` contain
   *distinguishable* components and may separate by instance where this does not.
   That is the experiment that would overturn the constraint above.

## Reproduce

```bash
py -3.12 train_slots.py \
  --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt \
  --num-slots 4,6,8,12 --epochs 150 --lr 8e-4

py -3.12 visualize.py \
  --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt \
  --mode slots --slot-checkpoint outputs/phase3a_slots/screw_board/slots_6.pt
```
