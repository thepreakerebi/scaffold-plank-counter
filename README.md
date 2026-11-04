# Scaffold plank counter

Counts steel scaffold planks in a photo of a stack, for the CountX iOS app's scaffold template.

The user marks each stack column with four corners, the same as CountX's range step. The counter then finds every plank end in that column and draws a numbered outline on it.

## How it works (classical, no training)

1. **Straighten** each marked column into a rectangle with a perspective warp, resampled to a fixed width. The same thresholds then work at any photo size.
2. **Hole pairs (count).** Every plank end has two punched holes. A row with two dark round blobs spread across the column is one plank.
3. **Seams (check).** The dark lines between plank ends give an independent count. If the two disagree, the column is flagged so the app can ask the user to review it.

## Run

```bash
uv sync
# put the five test images in data/raw/plank1.jpg … plank5.jpg (confidential, not in git)
uv run plank-count classical   # annotated images → outputs/classical/, table → outputs/classical.md
uv run pytest                  # synthetic tests, no photos needed
```

## Layout

| Path | What |
|---|---|
| `src/plank_counter/geometry.py` | quad warp and mapping back to the photo |
| `src/plank_counter/holes.py` | hole-pair rows (the count) |
| `src/plank_counter/seams.py` | seam profile (the cross-check) |
| `src/plank_counter/classical.py` | per-column counter + review flag |
| `annotations/columns.json` | the column corners marked for each test image |
| `annotations/ground_truth.json` | hand counts and the counting rule |
