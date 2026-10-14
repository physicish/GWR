# 07 — Background validation and time slides

**Chapter section 1.5.** Produces **Figure 1.3**.

Generating time slides cheaply is useful only if they give a reliable estimate
of the noise background. This stage tests whether false-alarm rates inferred
from shifted detector outputs predict the number of events found in
independent, unshifted data.

## The two data selections

Two **disjoint** selections of coincident O3a data:

| Purpose | GPS range | Duration |
|---|---|---|
| Zero-lag test | 1239392735 – 1240303634 | 5 days |
| Background | 1238166018 – 1238997902 | 5 days |

"Zero lag" means Hanford and Livingston retain their original relative timing.
Using disjoint stretches is the point: the background must not be built from
the data it is used to assess.

## Procedure

Data is processed in one-second windows at a 0.1 s stride, with per-detector
outputs stored. The background uses **100 relative time slides** at a spacing
of 5 s, so offsets run from 5 to 500 s — comfortably longer than both the
inter-detector light travel time and the input window.

Shifted pairs pass through the same ranking, thresholding and clustering as the
zero-lag data: triggers separated by no more than 0.35 s are grouped, and the
highest-ranked trigger represents each cluster. **Applying identical rules to
both analyses is essential**, otherwise a difference in event counts could
come from the processing rather than from the noise.

## Commands

```bash
# Zero-lag pass over the five-day test selection
python ../06_coincident_model/apply_coincident_transformer.py \
    --inputfile  o3a_zerolag_5day.hdf \
    --outputfile events_zerolag.hdf \
    --checkpoint ../06_coincident_model/runs/coincident/best_state_dict.pt \
    --step-size 0.1 --cluster-threshold 0.35

# Background: cache per-detector outputs, then 100 slides at 5 s spacing
python ../06_coincident_model/apply_coincident_transformer.py \
    --inputfile  o3a_background_5day.hdf \
    --outputfile events_background.hdf \
    --checkpoint ../06_coincident_model/runs/coincident/best_state_dict.pt \
    --step-size 0.1 --cluster-threshold 0.35 \
    --num-timeslides 100 --timeslide-step 5.0

python plot_far_validation.py \
    --test-file       events_zerolag.hdf \
    --background-file events_background.hdf \
    --test-duration-days 5 \
    --output figures/fig1_3_far_validation.png
```

## What you should get

After accounting for the overlap retained in each slide, the accumulated
background live time is **403.915 days**. The expected zero-lag count at
ranking threshold R is

```
N_expected(R) = T_0 * FAR(R)
```

with `T_0` the analysed zero-lag live time. Figure 1.3 plots the observed
cumulative count against inverse false-alarm rate, with 1–3σ Poisson bands
around the expectation. The zero-lag distribution should follow the time-slide
prediction and stay within those bands across the evaluated IFAR range.

Agreement means the time-slide procedure reproduces the coincident-noise tail,
and therefore that the false-alarm rates assigned to candidates can be trusted.

## The cost comparison

The 100-slide calculation takes about **30 minutes on a single CPU** — roughly
18 seconds per slide, or about 13.5 days of effective background per minute of
computation.

Running the same 100 slides with the **coherent** model takes about **10 hours**,
because every slide requires a fresh pass through the network. That twenty-fold
reduction in slide-stage wall-clock time is the entire reason the coincident
design was adopted, and it is what makes extended background estimation
practical.
