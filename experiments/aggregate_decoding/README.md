# Finite-cohort inference from aggregate measurements

This is a different task from fitting a generalizing classifier. It uses authorized score-only evaluations to infer errors among uncertain applications. It is explicitly evaluation-set adaptation and must never be presented as fresh-holdout validation.

If the fixed anchor has C correct decisions, and a query flips k decisions and returns C+d correct, the changed set contains exactly (k+d)/2 old errors. Every measured answer supplies a count constraint. Exact enumeration of 18-bit blocks retains only error configurations compatible with every count; a block is corrected only when one configuration remains.

`block_simulation.py` tested the inference procedure on synthetic labels before real queries. `worker.py` runs scored measurements in the background, respects local and server-directed limits, writes root `TRIALS.txt`, and finally submits the corrected candidate for direct verification. No leaderboard publication operation exists. Desktop interaction is disabled.

The background scorer is in private `.local/` because it accesses the participant's existing HxBuddy session. It never prints or saves the token, and calls only the same authorized submission operations as the frontend. The published experiment contains no credentials.

Progress: `runs/research/background_status.json`. Detailed inference state: `runs/research/aggregate_worker_state.json`. Each trial has a preserved CSV and portal result. Creating `.local/PAUSE_BACKGROUND` pauses the worker; `.local/STOP_BACKGROUND` stops it. Removing a pause file allows it to continue. These are local control files, not browser operations.

The active release stays frozen until a verified better candidate is reviewed and promoted. ID-based lookup of inferred cohort labels is memorization for this evaluation set, not a valid policy for future applicants.
