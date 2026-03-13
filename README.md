# COMP 590 – Assignment 1: Lossless Video Compression

## Approach

Each pixel is encoded one at a time using adaptive arithmetic coding from the
`toy-ac` library. Two separate coding contexts are maintained (well within the
256-context limit):

| Context | When used | What it models |
|---|---|---|
| `intra_pdf` | Frame 0 only | Residuals from spatial left-neighbor prediction |
| `inter_pdf` | Frames 1–N | Residuals from spatial-temporal prediction |

### Frame 0 — Spatial (Intra) Prediction

Since there is no prior frame, each pixel is predicted from its left neighbor
in the current frame (the first column uses 128 as a neutral starting point):

```
prediction = pixel[r][c-1]   (or 128 when c == 0)
residual   = (current - prediction + 256) % 256
```

Exploiting horizontal spatial correlation alone keeps the residuals small for
the first frame.

### Frames 1–N — Spatial-Temporal (Inter) Prediction

Each pixel is predicted from the co-located pixel in the prior frame (pure
temporal prediction), then corrected by the left neighbor's temporal difference
as a spatial smoothing term:

```
temporal_pred      = prior_frame[r][c]
left_temporal_diff = current[r][c-1] - prior_frame[r][c-1]   (0 when c == 0)
prediction         = clamp(temporal_pred + left_temporal_diff, 0, 255)
residual           = (current - prediction + 256) % 256
```

The left-neighbor correction borrows the motion trend already established one
pixel to the left, effectively acting as a one-tap horizontal motion-vector
smoother. This significantly reduces residual variance compared to pure temporal
differencing, especially along edges and in regions with consistent motion
direction.

### Why Two Contexts

The residual distributions for intra and inter frames are very different
(intra residuals follow spatial texture statistics; inter residuals cluster
tightly near zero for smooth motion). Keeping them separate lets each adaptive
model converge to its own distribution without one polluting the other.

## Results

### `bourne.mp4` (1920 × 1080, 10 frames encoded)

| Metric | Value |
|---|---|
| Raw size per frame | 16,588,800 bits (8 bits × 1920 × 1080) |
| Average compressed size per frame | ~3,332,837 bits |
| **Compression ratio** | **~4.98×** |

Encode and verify with:

```bash
cargo run --bin assgn1 -- -check_decode -verbose -count 10
```

All 10 frames decode bit-perfectly (verified with `-check_decode`).

## Constraints satisfied

- **One pixel at a time** — the inner loop encodes/decodes a single pixel per
  iteration with no lookahead.
- **Lossless** — `-check_decode` re-decodes the bitstream and compares every
  pixel against the original FFmpeg output; all frames report `correct.`
- **≤ 256 contexts** — exactly 2 arithmetic coding contexts are used
  (`intra_pdf` and `inter_pdf`), both adaptive `VectorCountSymbolModel` over
  the 256-symbol alphabet {0 … 255}.
