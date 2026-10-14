# PSTBench

Code for our paper **Protocol Matters: Auditing and Standardizing Photorealistic Style Transfer Evaluation**.

`pstbench` scores a folder of output images against a paired benchmark (content, style and ground truth for every pair) under a fixed protocol. Each result is saved with the protocol and the dataset revision.

## Install

```bash
git clone https://github.com/beomjin-ahn/pstbench && cd pstbench
pip install -e .            # Python 3.10+; a GPU (--device cuda) speeds up LPIPS
```

## Quickstart

```bash
pstbench demo                        # synthetic images, no dataset download (the first run fetches the LPIPS weights)
pstbench data fetch pst50            # PST50 at the pinned revision (about 510 MB); also: acetone (3.3 GB)
pstbench score --track pst50-paired --outputs path/to/outputs --name my-method
pstbench compare results/pst50-paired/identity results/pst50-paired/my-method   # paired tests against our Identity run
```

Output of `score` for the content images themselves (our `identity` run):

```
track: pst50-paired   protocol: pstbench   dataset: zrgong/PST50@6ed86e3   n=50
method    PSNR↑         SSIM↑            LPIPS↓           ΔE00↓         H-Corr(gt)↑      H-Corr(style)↑
identity  22.57 ± 1.19  0.9043 ± 0.0223  0.1437 ± 0.0207  10.10 ± 1.25  0.4092 ± 0.0641  0.2401 ± 0.0764

results/pst50-paired/identity/  (metrics.csv, summary.json, run.json)
```

`±` is half the width of the 95% bootstrap interval over pairs. Always include the Identity row (the content images, unchanged).

## Protocol

The PSTBench protocol of our paper, id `pstbench`. The metric and bootstrap parameters are in `src/pstbench/protocol/pstbench.yaml`.

- I/O: 8-bit sRGB PNG via Pillow, EXIF orientation applied; outputs with a non-sRGB ICC profile are converted to sRGB and noted in `run.json`.
- Resolution: the reference's size, no crop or downsampling. Outputs of another size are rejected unless `--allow-resize` is given (PIL bicubic; each resized pair is recorded in `run.json`).
- PSNR: sRGB, over all RGB values of a pair, per image, mean over pairs, capped at 100 dB.
- SSIM: Wang et al. 2004 via scikit-image, Gaussian window σ = 1.5, population covariance, valid region only, no downsampling, mean over channels.
- LPIPS: `lpips` 0.1.4, AlexNet, version 0.1, inputs in [−1, 1], at the reference size.
- ΔE00: CIEDE2000 (kL = kC = kH = 1) in CIELAB, sRGB matrix, D65, 2° observer, white point = row sums of the matrix. Mean over pixels.
- H-Corr: Pearson correlation of 256-bin RGB histograms, mean over channels, against the ground truth (ranked) and against the style (reported, not ranked).
- Aggregation: mean over pairs, 95% percentile bootstrap (10,000 resamples, seed 0). `pstbench compare RUN1 RUN2` runs paired Wilcoxon tests with Holm correction over the runs given, counting per-pair differences below 0.001 in PSNR, LPIPS and ΔE00 as ties (set in the protocol file), and refuses runs that differ in protocol, track or dataset revision.

`run.json` holds the protocol id, the dataset revision and any deviation from the protocol.

## Tracks

| Track | Content and style | Reference |
|---|---|---|
| `pst50-paired` | PST50 `content_709` + `paired_style` | `paired_gt` |
| `pst50-paired-log` | PST50 `content_log` + `paired_style` | `paired_gt` |
| `acetone-transfer` | AceTone-Bench `raw_image` + `ref_image` (1024 pairs) | `gt_image` |

An outputs folder holds one 8-bit RGB PNG per pair, named `<pair_id>.png` (`pst50-001` to `pst50-050` for PST50's `in1` to `in50`, `acetone-0000` onwards); other PNG files are refused and files of other types are ignored. The two PST50 tracks differ in input encoding and are ranked separately. Datasets are fetched from Hugging Face at a pinned revision into `~/.cache/pstbench/data` (or `$PSTBENCH_DATA`) and checked file by file; the log track needs `pstbench data fetch pst50 --splits content_log paired_style paired_gt`.

## Methods scored for our paper

All methods were run outside the kit and scored from their outputs; each has a `run.json` under `results/<track>/<name>/`, where the name is the lower-case method name (CAP-VSTNet: `capvstnet`; WCT²: `wct2`; ModFlows: `modflow`; SA-LUT: `salut_709` and `salut_log` on PST50, `salut` on AceTone-Bench).

| Method | Code | Weights (sha256 prefix) | Settings |
|---|---|---|---|
| Identity | – | – | The content image, unchanged. |
| Reinhard | PyPI `color-matcher` 0.6.0 | – | `ColorMatcher().transfer(src, ref, method="reinhard")` on RGB in [0, 1], rounded to 8 bits. |
| MKL | PyPI `color-matcher` 0.6.0 | – | `method="mkl"`, same inputs. |
| IDT | PyPI `python-color-transfer` 0.1.2a0 | – | `ColorTransfer().pdf_transfer(regrain=False)` on 8-bit BGR arrays, as the package expects. |
| HistMatch | `skimage.exposure.match_histograms` | – | Per-channel histogram matching to the style on RGB in [0, 1], rounded to 8 bits. |
| WCT² | github.com/clovaai/WCT2 @ `05d17c0` | `wave_encoder_cat5_l4.pth` `ebd55e09aa4f`, `wave_decoder_cat5_l4.pth` `85086ea2da11` | `option_unpool=cat5`, transfer at encoder, skip and decoder, alpha 1, no masks. Full resolution: content reflect-padded to a multiple of 16 and cropped back, style cropped to a multiple of 16. |
| CAP-VSTNet | github.com/linfengWen98/CAP-VSTNet @ `76a14b6` | `checkpoints/photo_image.pt` `e1b87d4d3cae` | Photorealistic mode, no masks. Full resolution: content reflect-padded to a multiple of 4 and cropped back, style long side ≤ 1280. |
| AdaIN | github.com/naoto0804/pytorch-AdaIN @ `47950d0` | `decoder.pth` `379ca41d59f3`, `vgg_normalised.pth` `804ca2835ecf` | alpha 1, no color preservation. Full resolution: content reflect-padded to a multiple of 8 and cropped back, style short side 512. |
| AceTone | github.com/martian422/AceTone @ `916393b` | HF `Vivre/AceTone-3B-PST-Preview` @ `9416b259`: `model-00001-of-00002.safetensors` `60d937b53d5a`, `model-00002-of-00002.safetensors` `71d87dd119d1`; `model/acetone-vqvae-d64.pt` `115e4e8c1476` from the code repository | Official `eval/predict_lut_ddp.py` settings (inputs 512 px wide, temperature 0.01, 128 new tokens), SDPA attention, the predicted LUT applied to the full-resolution content. Rec. 709 content on PST50. |
| SA-LUT | github.com/Ry3nG/SA-LUT @ `3e62f9c` | HF `zrgong/SA-LUT`, `epoch=100-step=4127466.ckpt.state.pt` `5ad3685cf6bb` | Official `inference_cli.py` in batch mode with the public checkpoint: LUT and context map predicted from 512×512 inputs, LUT applied to the full-resolution content. Run on Rec. 709 content (`salut_709`) and log content (`salut_log`). |
| NLUT | github.com/semchan/NLUT @ `66e271d` | `experiments/model.pth` `b8c9bbb765e0` | Official `inference_finetuning_image.py`, default settings: 40 fine-tuning iterations per pair from the released checkpoint at 256×256 (loaded with `strict=False`), LUT applied at full resolution. |
| IPST | github.com/RongLiu-Leo/IPST @ `fe78f02` | torchvision VGG19 `IMAGENET1K_V1` `dcbb9e9dad56` | Official `transfer.py`, image mode, seed 0 per pair. |
| ModFlows | github.com/maria-larchenko/modflows @ `e1884a0` | HF `MariaLarchenko/modflows_color_encoder`, `modflows_color_encoder_B6_dim_8195_iter_751001.pt` `78ed4c2cee34` | Official `run_inference()`, encoder B6, 8 flow steps, strength 1.0. |

`wct2_at912`, `capvstnet_at1280` and `adain_at912` are the same repositories at the sizes their scripts use by default (a 512-pixel short side for WCT² and AdaIN, a 1280-pixel long side for CAP-VSTNet; WCT² and AdaIN inputs resized to 912×512 beforehand, so that WCT²'s crop to a multiple of 16 and AdaIN's padding to a multiple of 8 change nothing), scored with `--allow-resize`.

## Reproducing our paper's numbers

`results/sensitivity/` and `results/sensitivity_acetone/` hold the per-pair scores of every method under every evaluation variant of our paper. Every statistic in our paper's tables and figures is computed from these files and the benchmark images.

They were written by `scripts/sensitivity.py` from the output folders (needs the `dev` extra). To score your own outputs under every variant: `python scripts/sensitivity.py --track pst50-paired my-method=path/to/outputs --out results/sensitivity` (`--device cuda` for LPIPS). `identity_log.csv` was written with the spec `identity_log=log:identity`, which reads PST50's log content in place of an outputs folder.

## License

MIT.
