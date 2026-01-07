# Syracuse AU/PSPI Per-Frame Analysis

## 1. Goals & Inputs
- Understand how the OpenFace AU-derived PSPI signal behaves within each Syracuse pain video.
- Compare per-frame/per-video PSPI statistics against self-reported pain levels to assess strong-clip-label vs. MIL assumptions.
- Data sources: `/data/Nbi/Syracuse/AU_from_openface/*.csv`, metadata `meta_with_outcomes.xlsx`, MARLIN repo (`/data/Nbi/Marlin/MARLIN`).

## 2. Pipeline Summary & Script Updates
1. **Preprocessing** (`--raw_au_dir /data/Nbi/Syracuse/AU_from_openface`): keep frames with `confidence ≥0.9` and `success == 1`, write `processed_*.csv`.
2. **Per-frame scoring** (`Syracuse/per_frame_pain_stats.py`):
   - Computes PSPI = AU04 + max(AU06,AU07) + max(AU09,AU10); optional AU43, custom weights.
   - Rolling smoothing (`--smooth_seconds`), default PSPI threshold from global quantile (`--quantile`).
   - Outputs per-video stats: `mean_score`, `var_score`, `p_high`, `high_frame_count`, plus new intensity metrics `max_score`, `quantile_90/95/99`.
   - Can join metadata (`--meta_excel`, `--meta_columns`) to produce `per_video_pain_with_labels.csv`.
3. Command for the final run:
   ```bash
   python Syracuse/per_frame_pain_stats.py \
     --raw_au_dir /data/Nbi/Syracuse/AU_from_openface \
     --confidence_threshold 0.9 \
     --score_type pspi \
     --smooth_seconds 0.1 \
     --quantiles "0.9,0.95,0.99" \
     --meta_excel /data/Nbi/Syracuse/meta_with_outcomes.xlsx \
     --meta_columns pain_level \
     --labeled_output_csv Syracuse/per_video_pain_with_labels.csv \
     --output_csv Syracuse/per_video_pain_stats_new.csv \
     --summary_json Syracuse/per_video_pain_summary_new.json
   ```

## 3. Global Statistics (0.2 s vs 0.1 s smoothing)
| File | Smoothing | within var | between var | within/between | avg `p_high` | Q1/Q2/Q3 of `p_high` | count `p_high≥0.6` | count `<0.2` |
|------|-----------|------------|-------------|----------------|-------------|----------------------|---------------------|--------------|
| `per_video_pain_stats.csv` | 0.2 s | 0.337 | 2.400 | 0.140 | 0.198 | 0 / 0.0017 / 0.170 | 17 | 74 |
| `per_video_pain_stats_0.1.csv` | 0.1 s | 0.351 | 2.400 | 0.146 | 0.198 | 0 / 0.0027 / 0.173 | 17 | 74 |

Interpretation:
- Most videos have very small high-pain frame ratios (`p_high` median ≈0, even for high self-report cases), yet between-video means still vary more than within-video variance.
- Reducing the smoothing window does not materially change the distribution, so results are robust against that hyperparameter.

## 4. Alignment with Self-Reported Pain
Using `per_video_pain_with_labels.csv` (73 videos with numeric pain levels):
- `corr(mean_score, pain_level) ≈ -0.07` ⇒ PSPI averages are **not positively correlated** with self-report.
- `corr(p_high, pain_level) ≈ +0.03` ⇒ near-zero correlation.
- Grouping by pain buckets (≤3, 3–6, >6) shows:
  - Mean `p_high`: 0.281 / 0.114 / 0.293 (medians ≈ 0 for all).
  - Mean PSPI: 2.97 / 2.05 / 2.67 (no monotonic trend).
- Highlighted anomalies:
  - **High `p_high ≥0.6` but low pain/outcome negative**: `IMG_0095`, `IMG_0037`, `IMG_0096`, `IMG_0006`, `IMG_0061`, `IMG_0060`.
  - **Low `p_high ≤0.2` yet outcome positive**: `IMG_0028`, `IMG_0031`, `IMG_0034`, `IMG_0046`, `IMG_0052`, `IMG_0055`, `IMG_0070`, `IMG_0072`, etc.
  - **High intensity (max≥6) but low pain≤3**: `IMG_0006`, `IMG_0008`, `IMG_0009`, `IMG_0015`, `IMG_0036`, `IMG_0040`, `IMG_0057`, `IMG_0098`, `IMG_0108`.
  - **Low intensity (max≤2) yet pain≥6**: `IMG_0027`, `IMG_0032`, `IMG_0047`, `IMG_0070`.

These mismatches are often tied to speaking/nurse-interaction scenes that drive AU04/AU07 without indicating high pain.

## 5. Intensity Metrics
`Syracuse/per_video_pain_stats_new.csv` (0.1 s smoothing):
- `max_score` mean = 4.26 (top: IMG_0045=9.57, IMG_0014=9.24, IMG_0098=8.38).
- `quantile_95` mean = 3.36; `quantile_99` mean = 3.78.
- Top vs bottom examples:

| Video | mean_score | max_score | quantile_95 | p_high |
|-------|-----------:|----------:|------------:|-------:|
| IMG_0045 | 4.07 | 9.57 | 5.23 | 0.457 |
| IMG_0014 | 4.77 | 9.24 | 6.37 | 0.619 |
| IMG_0098 | 6.47 | 8.38 | 6.58 | 1.000 |
| IMG_0012 | 0.20 | 0.59 | 0.38 | 0.000 |
| IMG_0047 | 0.80 | 1.14 | 0.69 | 0.000 |

Correlations with pain_level remain weak (`max_score ≈ -0.095`, `quantile_95 ≈ -0.03`), but these features help pinpoint “strong but sparse” peaks for MIL attention.

## 6. Peak Component Audit (Examples)
Top 5 peaks per video after 0.1 s smoothing:

| Video | Timestamp (s) | Smooth PSPI | AU04 | Dominant AU (6/7) | Dominant AU (9/10) |
|-------|---------------|-------------|------|-------------------|--------------------|
| IMG_0009 | 56.5 / 40.6 / 66.8 / 37.6 / 51.9 | ≈6.1–6.5 | 1.7–3.8 | AU07 (2.6–3.7) | Mostly AU09≈0 |
| IMG_0015 | 63.8 / 61.2 / 38.4 / 32.0 / 48.8 | ≈6.1–7.1 | 2.5–4.1 | AU07 ~2.1–2.4 | AU10 contributes 1.2–1.8 |

Takeaway: peaks consistently come from AU04 + AU07 (plus AU10 for video 15); other AUs stay low, so mislabeling likely stems from context (talking) rather than random AU spikes.

## 7. Recommendations
1. **Flag speaking segments** (high AU25/26) to avoid counting conversational expressions as pain.
2. **Use adaptive thresholds**: per pain bucket or per video (`mean + k·std`) instead of one global 80% quantile.
3. **Leverage intensity features** (`max_score`, `quantile_95`, `p_high`) as continuous inputs or MIL attention priors rather than binary heuristics.
4. **Manual QA** for the small set of conflicting cases to verify whether self-reports or AU readings are off.
5. **Documented artifacts**: `Syracuse/per_video_pain_stats.csv`, `_0.1.csv`, `_new.csv`, `_summary*.json`, `_with_labels.csv` should be kept with experiments for reproducibility.
