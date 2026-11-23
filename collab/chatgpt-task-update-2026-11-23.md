# ChatGPT implementation note — 23 November 2026

Branch: feat/pressing-defensive-moments-direction
Base: main at ef107581fe92f5412ae61930a8b0d9030f0f777b

## Implemented in this PR

### 1. Pressing proxy in scripts/team_report.py
Adds fields under each team in every period and the full-match aggregate:
- press_share
- press_eligible_samples
- pressing_samples

Definition:
- Denominator: wide-camera frames where the opponent is the possession owner and the ball is in the opponent's own third.
- Numerator: those denominator frames where the opponent is still the current holder and at least one of our outfield players is within 5 m of that holder.
- Frames with no current holder remain eligible in the denominator but cannot be counted as pressing frames.

The calculation uses the existing possession owner/holder timeline and the already-normalised home-view pitch coordinates. It is described as a tracking pressure proxy, not event-data PPDA.

**PPDA is intentionally deferred.** There is not yet a hand-tagged evaluation window for the current candidate generator, so candidate pass and defensive-action counts would be an unvalidated numerator/denominator.

### 2. Defensive player moments in scripts/build_player_moments.py
Adds the descriptor defensive_nearest_2 to existing per-track moments:
- The opponent must be the current owner and current holder.
- The ball must be in the defending team's own third, using home_attacks_right_by_period from resolved combined metadata.
- The two nearest outfield defenders are selected for each eligible frame.
- Direction-missing periods are skipped rather than guessed.
- Existing gap-merging, quality calculations, ok/review labels, and seven-column cut-list are retained. When defensive and near-ball observations overlap in one window, the events descriptor joins both types.

The tracking CSV and cut-list schemas do not change.

### 3. Direction fallback in team_report.infer_direction
When the first 10 seconds of a period do not yield a valid kick-off-position vote, scans up to the first 10 minutes for a contiguous centre-ball restart run. A vote requires repeated ball observations near the centre and clear outfield-team separation into opposite halves. It returns at most one strongest restart vote per period. Existing possession-shift votes and config precedence remain unchanged.

This is a fallback, not a guarantee: if the match has no clear centre restart and possession shift is weak, direction remains unknown.

## Tests added
- Press-share denominator/numerator and 5 m threshold.
- Delayed centre restart when the configured period begins in pre-roll.
- Defensive nearest-two selection, direction gating, and exact seven-column output.

CI result: pending GitHub Actions on the PR.

## Still waiting for Rizki's data
1. Shirt-number OCR check: needs the next Kaggle bundles for persiba_gresik_20261113 and persipa_persiba_20261117 with READ_NUMBERS=True (or DB with v_track_numbers). The previously uploaded PERSIBA–Persika bundle contains no number-reader output / track_map.
2. Throughput A/B: needs the same 10-minute chunk actually run with defaults, --pitch-every 2, and --imgsz 800; the previous bundle has no alternate runs or original master video to replay.
3. Candidate tuning: needs one hand-tagged window and evaluate_candidates.py JSON. No thresholds were tuned.

## Validation limits
I have inspected the current main code and added targeted tests, but have not run these tests locally against a checkout. Use the PR CI result as the execution check. Do not merge until CI is green and the definitions are reviewed.
