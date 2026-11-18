# Track A review handoff — 15 November 2026

This note is based on a fresh read of the current main branch files on 15 Nov 2026. Please verify each point against the latest main before editing because Track B is being developed on track-b/automatic-tracking-v1.

## Priority 1 — prevent silent data corruption or wrong joins

### 1. Downloader final-file resolution still conflicts with DECISIONS.md

File: src/download_match.py, function resolve_output().

Current order checks requested_downloads[0].filepath/_filename before the top-level info["filepath"]. The requested download entry can be an intermediate video/audio component, while the top-level path represents the merged/post-processed result. The fallback also returns hits[0] when there are no valid video extensions, which can select a .part or unrelated file.

Requested fix:
- Prefer top-level info["filepath"], then top-level _filename.
- Only then consider requested-download candidates, and only accept a real final media file.
- Reject .part, .ytdl, temporary download and audio-only files.
- Add mocked tests for merged output, absent top-level filepath, stale .part, and no valid output.
- This corrects the mismatch with the 15 Nov DECISIONS entry saying the merged filepath is preferred.

### 2. CSV import must validate match_id before mutating active events

File: tagger/index.html, CSV import handler near const lines=parseCSV(String(rd.result)).

The import does not read or validate incoming match_id. A CSV exported for another match can therefore be added to the currently open match. The current row loop also appends rows as it goes, so a late validation failure could leave a partial import.

Requested fix:
- Read the CSV match_id column before mapping rows.
- If present, require every non-empty match_id to equal the active match's immutable ID.
- For legacy CSVs without match_id, ask for explicit confirmation and show a warning.
- Validate all rows into a temporary array, then append once validation is complete.
- Add fixtures for matching IDs, mismatched IDs, missing IDs and malformed rows.

### 3. Assist candidate search can cross an opposition turnover

File: tagger/index.html, candidateAssist().

The current before array filters to e.team===S.team before looking for possession breaks. That hides opposition events from the scan. Example: home completed pass → away interception/recovery → home regains → home shot. The old home pass can remain eligible even though the opponent had taken possession.

Requested fix:
- Scan the complete event stream in match-time order, not a pre-filtered same-team list.
- Reset/break the candidate pool at credible opposing possession-control events (e.g. successful interception/tackle, recovery, or other clearly possession-changing event).
- Keep the assist rule transparent: a completed chance-creating pass can be selected when receiver/shooter evidence supports it; ambiguous cases should not be forced.
- Test the turnover sequence above and an uninterrupted pass → layoff → shot sequence.

## Priority 2 — make failure states honest

### 4. Local delete errors are swallowed

File: tagger/index.html, Store.deleteMatch().

The localStorage branch catches and ignores every exception. The caller then reaches its success toast, so it can say "Match deleted" even when browser storage was not changed.

Requested fix: let errors propagate, as the UI handler already has a try/catch that can show failure. Add a quota/security-denied test if practical.

### 5. Corrupted-load state should be visibly read-only

File: tagger/index.html, openMatch(), markDirty(), tag controls.

When a load fails, S.loadFailed=true and markDirty() declines to save. However, the interface remains interactive, so the user can still tag events in memory and see no obvious read-only state; those new events will not persist.

Requested fix:
- Render a prominent "READ ONLY — saved events could not be loaded" banner.
- Disable event tagging/edit/delete/import controls until the match is reloaded or a recovery action is taken.
- Keep CSV export/recovery available where possible.
- Do not show ordinary "Saved" status for this state.

## Integration changes now present on the Track B branch

- DATA_CONTRACT.md clarifies event side vs event display-name team, and tracking team=home|away|unknown; clarifies video_seconds as presentation time / PTS where available, with frame/FPS only as a CFR fallback.
- src/heatmap_from_positions.py reads home_attacks_right_by_period from the tracking metadata file and refuses to infer a missing extra-time direction. The old --home-dir R/L remains as a legacy fallback.
- Please carry these changes into main when reviewing/merging Track B.

## Methodology point to confirm

### 6. xT includes "From Kick Off" in OPEN_PATTERNS

File: scripts/fit_xt.py, OPEN_PATTERNS.

The selected list includes From Kick Off, although kick-offs are restarts. This may be an intentional exception to include the first attacking phase after a restart, but it is not strict "open play only" in the literal sense.

Please either:
- Exclude From Kick Off and refit the grid, or
- Keep it and describe the method as "open-play-like sequences plus kick-offs", explaining why.

No need to reopen other xT choices already recorded in DECISIONS.md unless a test contradicts them.
