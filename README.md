# dots-tts-mlx

A pure-[MLX](https://github.com/ml-explore/mlx) port of [`rednote-hilab/dots.tts`](https://github.com/rednote-hilab/dots.tts) — multilingual zero-shot voice-clone text-to-speech, running natively on Apple Silicon.

dots.tts is a **2B-parameter, fully continuous, end-to-end autoregressive flow-matching** TTS model (the `dots.tts-soar` SCA checkpoint). Unlike discrete-codec TTS models that warm up from a quantized token stream, dots.tts is continuous AR — so the **first patch is already a crisp utterance onset**, with no warm-up mumble at sample 0. It clones a voice from a short reference clip and synthesizes into 24 languages.

This repo is a clean-room MLX reimplementation of the runtime: no PyTorch calls in the inference path, gated per-stage against the original PyTorch model.

## Scope — what this is / isn't

This is a **converted-weight MLX inference runtime** for the `dots.tts-soar` (SCA) checkpoint. It deliberately does **not** replicate upstream's full surface. It **is**:

- a from-scratch MLX port of the dots.tts inference math, numerically gated against the original PyTorch model;
- a CLI + Python API that synthesizes from a **local, already-converted** weights directory.

It is **not** a drop-in replacement for the upstream package. In particular, this runtime does **not**:

- **auto-download by HF repo id** — you point it at a local converted directory (see [Weights](#weights)); there is no hub fetch baked into the runtime;
- **auto-detect language or resolve language names** — pass an **explicit uppercase ISO code** (e.g. `HI`, `ES`, `EN`), not `auto_detect` or a spelled-out name;
- **normalize text** — there is no `--normalize-text`; feed already-normalized input;
- **support random / no-reference sampling** — a reference clip (`--ref-audio`) is required for the voice clone;
- **ship a Gradio app** — CLI + Python API only;
- **do fine-tuning or training** — inference only.

If you need any of those, use the upstream project: [code](https://github.com/rednote-hilab/dots.tts) · [model](https://huggingface.co/rednote-hilab/dots.tts-soar).

## What it is

- **Architecture:** Qwen2.5-1.5B-Base LLM backbone (BPE text, no phonemes) → AR flow-matching DiT head → 48 kHz AudioVAE with a BigVGAN-style causal decoder. A frozen CAM++ x-vector conditions the speaker identity.
- **Zero-shot voice clone:** one reference wav (+ optional transcript) clones the voice; cross-language transfer works from a single reference.
- **Clean onsets:** continuous AR means no discrete-token warm-up — the clip opens on a real word.

## Headline results

Cloned into 5 languages from **one English reference** (`--num-steps 10 --guidance-scale 1.2 --speaker-scale 1.5`, bf16 runtime), scored with MLX-Whisper WER + CAM++ speaker-SIM:

| Lang | Tier | WER | speaker-SIM |
|------|------|-----|-------------|
| EN | ship | **0.000** | 0.830 |
| DE | ship | **0.000** | 0.794 |
| ES | ship | **0.000** | 0.781 |
| FR | ship | **0.000** | 0.719 |
| HI | preview | **0.105** | 0.784 |

EN/DE/ES/FR are ship-tier (0.0 WER on short, clean, in-domain sentences — better than the dots.tts paper's quoted ~1–3.5% for these targets). Hindi at 0.105 WER is **preview-tier**: fully intelligible, with only minor diacritic/spelling slips that are phonetically correct. SIM 0.72–0.83 is strong; the English self-clone (0.830) is the ceiling, with cross-language transfer sitting just below as expected.

24 languages are supported overall (the model's full coverage); the matrix above is the validated subset. See [How it was ported / parity](#how-it-was-ported--parity) below for the per-stage gates and methodology.

## Install

Requires Python ≥ 3.10 on Apple Silicon (MLX is Metal-only).

```bash
# quickest — install the published release directly:
pip install "git+https://github.com/sb1992/dots-tts-mlx.git@v0.1.0"

# or, for development (editable):
git clone https://github.com/sb1992/dots-tts-mlx.git
cd dots-tts-mlx
pip install -e .
```

The runtime deps are `mlx`, `mlx-lm`, `numpy`, `soundfile`, `tokenizers`. The `dots-tts` console command and `python -m dots_tts_mlx.cli` are both installed.

For **weight conversion** and the **dev parity oracle** (both use PyTorch), install the extra:

```bash
pip install -e '.[oracle]'   # torch, transformers, torchdiffeq, safetensors, librosa, torchaudio
```

Two distinct workflows use this extra — don't conflate them:

- **Weight conversion** (`python -m dots_tts_mlx.convert`) needs only `torch` + `safetensors` + `numpy` (a subset of `[oracle]`). It does **not** need the upstream package.
- **Regenerating parity fixtures** (`tools/oracle.py`) needs the full `[oracle]` extra **and** the upstream `dots_tts` package, which is **not on PyPI** — install it separately from source:

  ```bash
  pip install -e /path/to/dots.tts
  # or: pip install "git+https://github.com/rednote-hilab/dots.tts"
  ```

`ffmpeg` is needed only if you use `--speed` (post-hoc time-stretch).

## Weights

The MLX runtime needs converted weights — convert the original checkpoint once:

```bash
# 1. Download the original checkpoint from Hugging Face (~9 GB of safetensors).
huggingface-cli download rednote-hilab/dots.tts-soar --local-dir weights/dots_tts_src/dots.tts-soar

# 2. Convert HF -> MLX fp32 safetensors (needs the [oracle] extra for torch).
python -m dots_tts_mlx.convert \
    --src weights/dots_tts_src/dots.tts-soar \
    --out weights/dots_tts_mlx
```

The converter folds the vocoder's `weight_norm` (80 pairs), passes the speaker BN buffers through, extracts `latent_stats`, and copies the config + tokenizer alongside the weights. Output is **fp32 MLX safetensors (~9 GB on disk)**; at runtime the loader casts to **bf16 (~10 GB resident)**. fp32 parity runs use roughly 2× that.

> Note: only the converted artifacts are needed to run; the original checkpoint + `[oracle]` extra can be removed afterward.

### Quantized weights (smaller download)

The ~9 GB fp32 directory can be shrunk with `python -m dots_tts_mlx.quantize`, which quantizes **only
the Qwen2.5 LLM trunk** (70% of the weights) and keeps the precision-sensitive flow-matching DiT, the
BigVGAN vocoder, and the CAM++ speaker at bf16. The output is a self-contained directory that loads
exactly like the fp32 one — the loader auto-detects the `quantization` block in `config.json`, so no
flags change at inference time.

```bash
python -m dots_tts_mlx.quantize --src weights/dots_tts_mlx --out weights/dots_tts_mlx_int4 --bits 4
#   --bits 16 → bf16 (no quantization)   --bits 8 → int8-LLM   --bits 4 → int4-LLM   (--group-size 64)
```

Validated on a 5-language clone (EN/DE/ES/FR ship + HI preview) from one English reference:

| Variant | Download | WER (EN/DE/ES/FR / HI) | speaker-SIM |
|---------|----------|------------------------|-------------|
| fp32 | ~8.9 GB | — | — |
| bf16 (runtime dtype) | ~4.5 GB | 0.00 / 0.105 | 0.71–0.83 |
| int8-LLM | ~3.1 GB | 0.00 / 0.105 | 0.69–0.82 |
| **int4-LLM** | **~2.4 GB** | **0.00 / 0.105** | 0.68–0.80 |

WER is identical at every precision; speaker-SIM differences sit within run-to-run measurement noise.
**int4-LLM (~2.4 GB, −73%) is the recommended download**, with int8 as a conservative fallback. The
quantizer requires only `mlx` + `mlx-lm` (no torch).

## CLI usage

```bash
dots-tts \
    --text "Hello, this is a quick test of the on-device voice." \
    --ref-audio path/to/reference.wav \
    --ref-text "transcript of the reference clip" \
    --language EN \
    --out-path outputs/dots_tts \
    --out-prefix my_clone
# -> outputs/dots_tts/my_clone_000.wav  (48 kHz)
```

Key flags:

- `--ref-audio` (required) — the voice to clone. `--ref-text` is the reference transcript; **omit it for an x-vector-only clone** (no prompt transcript).
- `--language` — uppercase ISO code (`EN` / `DE` / `ES` / `FR` / `HI` / …). Default `None` = no language tag.
- `--num-steps 10` `--guidance-scale 1.2` `--speaker-scale 1.5` `--seed 42` — flow-matching sampler knobs (defaults are the validated ship config).
- `--speed 1.0` — playback tempo via pitch-preserving ffmpeg `atempo` (`<1` slower, `>1` faster). dots.tts is a self-pacing AR model with **no native rate control**, so this is a post-hoc time-stretch applied after onset-trim.
- `--trim-onset` / `--no-trim-onset` — `--trim-onset` is **on by default**: it removes the fixed ~50–150 ms BigVGAN vocoder onset transient (a soft "hhh"/breath at sample 0) via an energy gate + 10 ms anti-click fade. `--no-trim-onset` keeps the raw vocoder output verbatim.

## Python API

```python
import mlx.core as mx
from dots_tts_mlx.loader import from_pretrained

model = from_pretrained("weights/dots_tts_mlx", dtype=mx.bfloat16).model

out = model.generate(
    text="Hello from MLX.",
    prompt_audio="reference.wav",
    prompt_text="transcript of reference.wav",   # or None for x-vector-only clone
    language="EN",
    num_steps=10,
    guidance_scale=1.2,
    speaker_scale=1.5,
    seed=42,
)

wav = mx.array(out["audio"]).astype(mx.float32)   # mono float32
sr = out["sample_rate"]                            # 48000
```

## How it was ported / parity

Every stage was gated numerically against the original PyTorch model (a dev-only oracle, `tools/oracle.py`, dumps reference fixtures) before any behavioral test. Each gate uses a manual high-precision fp32 reference matmul so a single component's parity is isolated from the runtime's reduced-precision matmul:

| Stage | Metric | Result |
|-------|--------|--------|
| AudioVAE decode | PSNR vs torch | 55.67 dB |
| AudioVAE encode | max-abs | 2.3e-4 |
| CAM++ x-vector | cosine | 0.99999988 |
| Attention (rotary) | max-abs | 4.77e-7 |
| DiT velocity field | cosine | 0.9999962 |
| Semantic patch encoder | cosine / max-abs | 0.9999995 / 4.69e-4 |
| LLM hidden (Qwen2.5) | cosine | 0.99999970 |
| Flow solver (true-fp32 floor) | max-abs | 1.4e-4 |

**The tf32 finding — why the end-to-end test is behavioral, not sample-exact.** MLX's fast matmul rounds fp32 operands to ~tf32 (10-bit mantissa) on the GPU. The per-stage gates sidestep this with an explicit high-precision path. But the euler ODE in the flow solver *amplifies* the per-step DiT matmul tf32 floor (fast-path max-abs 0.577 vs the true-fp32 floor of 1.4e-4), so across 10 integration steps × N patches the trajectory diverges enough that the waveform does **not** sample-align with the PyTorch golden. This is a runtime-precision property, not a port bug — sample-exact e2e parity isn't reachable on this hardware path. So the e2e gate is **behavioral**: WER 0.0 / SIM 0.829 / finite / right duration / clean onset confirm the output is *correct*. The component-level numerics prove the math; the behavioral gate proves the product.

Tests live in `tests/` (pytest). They skip cleanly when the converted weights or PyTorch fixtures are absent. Regenerate fixtures with `tools/oracle.py` under the `[oracle]` extra **plus** the upstream `dots_tts` package installed from source (`pip install -e /path/to/dots.tts`) — see [Install](#install).

## Limitations

- **Hindi is preview-tier** (0.105 WER) — correct and usable, but slightly higher WER on the low-resource tail, as the paper predicts. EN/DE/ES/FR are ship-tier.
- **Memory:** ~10 GB resident at bf16 (dominated by the resident Qwen2.5 backbone + DiT); fp32 parity runs land ~2× that.
- **No native speech-rate knob** — dots.tts is a self-pacing AR model; pacing is controlled post-hoc via `--speed` (ffmpeg time-stretch).
- **Apple Silicon only** — MLX is Metal-only.
- While the runtime makes **no torch calls**, `torch`/`transformers` may be resident transitively via `mlx-lm`'s tokenizer utilities. The inference math is pure MLX.

## Responsible use

This runtime performs **zero-shot voice cloning** — it can reproduce a person's voice from a few seconds of reference audio. That capability carries real risk of misuse. By using this software you agree to use it responsibly. Mirroring the upstream [dots.tts-soar risks guidance](https://huggingface.co/rednote-hilab/dots.tts-soar):

- **No impersonation, fraud, or disinformation.** Do not use cloned voices to impersonate real people without authorization, to commit fraud or social engineering, to evade voice-based authentication, or to produce misleading or deceptive content.
- **Consent for reference audio.** Only clone a voice you own or for which you have the speaker's explicit, informed consent. Respect the rights of voice owners and applicable privacy / publicity / data-protection laws in your jurisdiction.
- **Disclose AI-generated audio.** Clearly label synthesized speech as AI-generated wherever it is published or shared, so listeners are never misled about its origin.
- **Watermark + detect downstream.** You are encouraged to apply audio watermarking to generated output and to deploy synthetic-speech detection in any pipeline that ingests it, to support provenance and abuse mitigation.

The authors and contributors disclaim responsibility for misuse. Comply with all applicable laws and with the upstream model's license and usage terms.

## Attribution + licenses

This is a derivative port. The original model, its backbone, and the components it builds on are each independently licensed (all verified):

- **dots.tts** (`rednote-hilab/dots.tts-soar`) — **Apache-2.0**. The model, weights, and the *dots.tts Technical Report* (2026) are by the dots.tts team at rednote-hilab. [Model](https://huggingface.co/rednote-hilab/dots.tts-soar) · [Code](https://github.com/rednote-hilab/dots.tts) · [Demo](https://rednote-hilab.github.io/dots.tts-demo/)
- **Qwen2.5-1.5B-Base** (LLM backbone) — **Apache-2.0** ([Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B)).
- **CAM++ / 3D-Speaker** (speaker x-vector encoder) — **Apache-2.0** ([modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)).
- **BigVGAN** (the vocoder/decoder architecture style) — **MIT**, © NVIDIA ([NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN); parts adapted from [hifi-gan](https://github.com/jik876/hifi-gan), MIT).

The MLX port code in this repository is licensed **Apache-2.0** (see [LICENSE](LICENSE)). You must comply with the upstream licenses for the model weights and any redistributed components.

### Credit

Full credit to the **dots.tts team at rednote-hilab** for the model, the SCA post-training, and the open release. This repo only re-expresses their runtime in MLX; the research, training, and weights are theirs.
