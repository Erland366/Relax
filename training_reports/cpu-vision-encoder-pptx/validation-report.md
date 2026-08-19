# CPU vision encoder slide validation

- Deck: `cpu-vision-encoder-project-update-2026-08-19.pptx`
- Result: **PASS**
- Slide count: **4**
- Canvas: **16:9, white, monochrome authored content**
- Editable objects: **native PowerPoint text, shapes, lines, and table**
- Speaker notes: **present on all four slides**
- Text-object overlap check: **PASS**
- Renderer: **not available in this environment; no native rendered visual approval claimed**

## Static findings

- No static validation findings.

## Verified result table entries

- CPU/native token match: `8/8`; maximum log-probability drift: `1.43e−6`.
- Peak demand / one-core supply: `5.86 / 17.43 images/s` (`2.98×`).
- Batch-eight gain: `7.96%`, below the `20%` eligibility gate.
- Tier 2: request bytes `−99.82%`, serialization `−77.94%`, p95 RTT `−29.30%`.
- Rollout wall: `1.344 → 1.261 s` (`−6.17%`), below the `10%` adoption gate.
