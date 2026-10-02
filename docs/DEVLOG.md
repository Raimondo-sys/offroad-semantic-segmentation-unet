# DEVLOG — preparing the repository for publication

This log records how the original course deliverable was turned into this repository: what was kept, what was removed, where every figure and number comes from, and which inconsistencies were found.

---

## 2026-10-01/02 — source material

The project arrived as a single Google Drive export (`ML_Project_Work-…zip`, 204 MB), containing:

| File | Size | Fate |
|---|--:|---|
| `TrainingGroup32.ipynb` | 60 KB | Published as `notebooks/train.ipynb`: first byte-identical, then bugs fixed after submission (see the 2026-10-02 section) |
| `test.ipynb` | 58 KB | Published as `notebooks/evaluate.ipynb`, code cells unchanged, outputs removed |
| `ReportGruppo32.pdf` | 1.5 MB, 22 pages | Not published: the title page lists student IDs and e-mail addresses. 16 figures extracted to `assets/` |
| `PresentazioneGruppo32.pptx` | 6.4 MB | Not published: over 5 MB, the last slide lists student IDs and e-mails, the file metadata contains a full name |
| `MembriGruppo32.txt` | 138 B | Not published: student IDs and e-mails |
| `model.pth` | 98 MB | Not published: PyTorch pickle archive (`Stage2FineTunedDiceCE_model/`), too large and unsafe to load from an untrusted source |
| `modelstages/` | empty | Not published |
| `train/archive/train/` | 104 MB, 743 samples | Not published (dataset) |

There was no `.git` folder, `__pycache__` or other checkpoint.

---

## What the deliverable was

- **Provided by the course**: the 931-image training split of the Yamaha-CMU Off-Road Dataset (renumbered, masks as index PNGs) and the `test.ipynb` template:
  - `#IMPLEMENT HERE` cells for `load_model()` and `predict()`;
  - an evaluation loop marked *"DO NOT MODIFY"*, used by the instructors on a private test set ("Final competition score").
- **Constraints**: no external data, dynamic augmentation allowed, Google Colab, at most 5 GB GPU for training and 4 GB for testing.
- **Delivered by the group**: the cleaned dataset (743 images), the training notebook, `model.pth`, the report and the slides. The files are dated 2025-07-09 (20:13–20:31) and the report is dated 10 July 2025. `test.ipynb` was saved later, on 2025-07-15.

---

## Reconstructing the history of the code

The training notebook has **no outputs at all** (every cell has `execution_count = None`). Its code also differs in several places from the run shown in the report:

| Evidence | Run in the report | Published notebook |
|---|---|---|
| Message after saving the best model (phase 4) | `Best model (finetune) saved.` | `Best model saved.` |
| Checkpoint name | `Stage2FineTunedDiceCE_model` (inside `model.pth`) | `FourthStageModel.pth` |
| Plot titles | separate *Training/Validation Loss* and *Validation mIoU* figures | one combined *Loss & mIoU* figure |
| Phase-3 learning rates (enc / dec / head), weight decay | screenshot on p. 14: 1e-6 / 1e-5 / 1e-4, wd 1e-5, `max_class_weight = 10.0` | 1e-5 / 1e-4 / 1e-4, wd 1e-5, `max_class_weight = 20.0` |
| Phase-3 length | 20 epochs | up to 30, early stopping after 7 epochs without improvement |

The notebook metadata contains both a `colab` block (T4 GPU) and a `kaggle` block listing 9 dataset versions.

The cell metadata also records the **last execution** of this notebook, on Kaggle on 2025-07-09:
- cells 0–38 ran between 12:26 and 12:28 UTC;
- the phase-1 training loop (cell 39) ran from 12:28:20 to 12:39:59, about 12 minutes, far less than 30 epochs at 544 × 1024;
- every later cell is marked idle at the same instant without an `execute_input` timestamp, i.e. it was queued but never ran.

So the published notebook, in its final form, was last run as a phase-1 run that was interrupted. It was never run end to end, which is consistent with the `NameError` in phase 3 going unnoticed.

The most likely explanation is that the project was trained in several sessions and notebook versions (Colab and Kaggle; on Kaggle, checkpoints are typically passed between sessions as datasets). The versions were then merged into one tidy notebook for submission, and its outputs were cleared. The undefined `train_losses_fn` lists are typical of such a merge: in the original version a cell defined them.

`test.ipynb` contains a single **interrupted** run:
- runtime TPU with a CPU-only PyTorch, about 7 s per image;
- stopped by hand after 10 of 743 images;
- run on the training folder, as the course template does by default.

It was a check that `load_model()` and `predict()` work, not an evaluation. Its printed "score" is meaningless, because the 733 unprocessed images count as zero.

Changes made to `evaluate.ipynb` (code cells untouched, verified cell by cell against the original):
- outputs and execution counts removed, for the reason above;
- stored widget state removed (download progress bars);
- Colab per-cell metadata (`executionInfo`, `outputId`, `colab`) removed: `executionInfo` contained the Google account's display name and **numeric user ID** and the execution timestamps. The final scan found it; the first review had not.
- later, for consistency with `train.ipynb`: notebook-level `colab` and `accelerator` (TPU) blocks and the remaining cell ids removed. Only `kernelspec` and `language_info` are left.

---

## Decision: publish with the report's results, code unchanged

Two options were considered: publishing now with the report's results honestly labelled (A), or re-running the training on Colab to obtain reproducible numbers (B). Option A was chosen. Consequences:

- The code of both notebooks is published as submitted. Known bugs are documented in the README ("Known issues") instead of fixed, so that the notebook stays the one that was submitted.

**Update, 2026-10-02.** Before the push, the bugs of `train.ipynb` were fixed after all (see the next-to-last section). The results still come from the report. The corrected notebook has not been re-run on a GPU.
- Every result in the README carries its source: re-computed, printed output, read from a plot, or report text only.

---

## What was verified, and how

### Re-computed from the data (verifiable)

The split function and `compute_class_distribution` were run **verbatim**: their source was read from the notebook cells and executed, not re-typed. They ran on the 743 delivered samples (Python 3.14, numpy 2.5.3, Pillow 12.3.0, iterative-stratification with scikit-learn 1.9.1):

- split sizes: **531 train / 133 validation / 79 test**, no overlap between subsets;
- pixel share and number of images containing each class, overall and per split (README tables).

The per-split pixel distributions match the three histograms in the report (pp. 8–9; e.g. train background 3.1%, validation smooth trail 17.0%, test rough trail 17.6%). The report's curves were therefore produced on this exact split.

All 743 masks use the same palette (index → RGB colour table in the README).

### Taken from a printed output in the report (exact)

The screenshot of the last epoch of phase 4 (report p. 21, `assets/phase4_final_epoch_output.png`) shows:
- per-class IoU `[nan | 0.615980 | 0.544352 | 0.536918 | 0.350582 | 0.459904 | 0.292843 | 0.795712 | 0.916415]`;
- `Val IoU: 0.5641`;
- learning rate 0.000010.

The mean of the eight non-NaN values is 0.56409, consistent with the printed mIoU.

### Read from the report's curves (approximate)

The end-of-phase validation mIoU of phases 1–3 (≈ 0.43, ≈ 0.48, ≈ 0.53) and the number of epochs per phase (30, 48, 20, 30) were read from the x and y axes of the figures.

### Report text only (not verifiable)

| Claim | Note |
|---|---|
| Test mIoU "0.5" | Would come from the test cell of the training notebook; output not preserved |
| Inference time "3.5 s", called "average" in the report | `measure_inference_time` returns the **total** time over the test loader, so this is most likely the time for all 79 test images |
| GPU peak 3.8 GB (frozen encoder) and 4.7 GB (fine-tuning) | Not measured in the code; presumably read from the Colab resource monitor |
| 931 → 743 cleaning, 566-image second attempt | Consistent with the 743 delivered samples; the discarded IDs are not available |

---

## Inconsistencies between notebook, report and slides

| Item | Notebook | Report | Slides |
|---|---|---|---|
| Phase 3: encoder / decoder / head learning rate | 1e-5 / 1e-4 / 1e-4 | 1e-5 / 1e-4 / 1e-4 (text); 1e-6 / 1e-5 / 1e-4 (code screenshot, p. 14) | 1e-6 / 1e-5 / 1e-4 |
| Phase 4: head learning rate | 5e-5 | 1e-5 | 5e-5 |
| Phase 4: weight decay | 1e-4 | 1e-5 | — |
| Scheduler monitors | submitted: `val_iou` in phase 1, `val_loss` in phases 2–4 (with `mode='max'`); published: `val_iou` in every phase | validation mIoU | — |
| Augmentation | no cropping | no cropping | mentions "random cropping" |
| Inference time | total over the test loader | "average inference time 3.5 s" | — |
| Title | — | "Autoencoder based on board image segmentation" (course title) | the model is a U-Net, not an autoencoder |

The README uses the notebook's hyperparameters and states that the exact values of the run are uncertain.

---

## Known issues in the submitted code

1. `train_losses_fn`, `val_losses_fn`, `val_ious_fn`, `val_class_ious_fn` are never defined: phases 3 and 4 raise `NameError`.
2. `ReduceLROnPlateau(mode='max')` is stepped with `val_loss` in phases 2–4. The final-epoch screenshot still shows the initial decoder learning rate (1e-5), which this code would have reduced. The original run probably stepped on `val_iou`, as the report states.
3. `max_class_weight` is defined but never passed to `get_class_weights` (the cap is always 20).
4. No random seeds for PyTorch / NumPy. Only the split is deterministic.
5. The augmentation preview (cell 10) reads from `cleandataset/`, which is not part of the delivery. `plot_augmentation_steps` is defined twice.
6. `plot_training_curves` labels classes with an off-by-one legend ("Classe 1" = background) and writes every phase to the same `training_curves.pdf`.
7. The validation metric is a dataset-level IoU (intersections and unions summed over images). The course's evaluation computes a per-image IoU, so the two are not comparable.
8. In `evaluate.ipynb`, `test_dir` points to the training folder (course template default), and `load_model()` downloads the ImageNet encoder weights before overwriting them with the checkpoint.
9. `get_loss_fn(name, class_weights, device="cuda")` always moves the class weights to `cuda` and ignores the notebook's `device` variable. Training therefore requires a CUDA GPU. Found by the CPU smoke test on 2026-10-02.

Issues 1–6 were fixed on 2026-10-02 (next section). Issues 7 and 8 are left as they are: 7 because every reported result uses that metric, 8 because it is the instructors' template. Issue 9 was found later the same day by the CPU smoke test and fixed after it was approved (last row of the table below).

---

## 2026-10-02 — fixes after submission (`train.ipynb`)

Constraint: architecture, losses, phases and hyperparameters unchanged. The fixes were applied by a script that checks that each original line occurs exactly the expected number of times before replacing it. Afterwards the whole notebook was compared with the original cell by cell. Changes, by cell index (0-based):

| Cell | Change | Reason |
|--:|---|---|
| 51, 56 | The curve-list initialization of phases 3 and 4 now creates `train_losses_fn`, `val_losses_fn`, `val_ious_fn`, `val_class_ious_fn` instead of re-creating the unused `train_losses`… | Issue 1: the loops append to the `_fn` lists, which were never defined (`NameError` at the first fine-tuning epoch). Each fine-tuning phase now gets its own curves, like phases 1 and 2 |
| 38 | `get_class_weights(train_ids, dataset_root, max_w=max_class_weight)` | Issue 3: the variable was ignored. The notebook's value is 20, the same as the function's default, so the weights do not change |
| 44, 51, 56 | `scheduler.step(val_loss)` → `scheduler.step(val_iou)` | Issue 2: the scheduler is created with `mode='max'`. Phase 1 already stepped on `val_iou`, the report says the scheduler followed the validation mIoU, and the final-epoch screenshot shows no learning-rate reduction |
| 9, 10 | Cell 9 now holds the single `plot_augmentation_steps(image, transformations)` (the version that cell 10 actually called). The unused `plot_augmentation_steps(image, transform)` is removed. Cell 10 reads sample `0022` from `DATA_ROOT` (or the first sample if `0022` does not exist) | Issue 5: duplicate definition; `cleandataset/` was not part of the project |
| 34 | The per-class curves skip index 0 and use `CLASS_NAMES[cls]` as labels. The colours stay tied to the class index, so they match the report's figures | Issue 6: legend shifted by one, background plotted as an empty line |
| 39, 44, 51, 56 | `save_path="training_curves_phase{1..4}.pdf"` | Issue 6: every phase overwrote `training_curves.pdf` |
| 2 | The Drive mount is wrapped in `try … except ImportError` | Lets the notebook run outside Colab |
| 3 | `PROJECT_ROOT` (environment variable, Drive path as default) → `DATA_ROOT`, `CHECKPOINT_DIR` (created if missing). `dataset_root = DATA_ROOT` keeps the rest of the code unchanged | Hard-coded Drive paths in 9 places |
| 39, 41, 44, 48, 51, 53, 56 | Checkpoint paths built with `os.path.join(CHECKPOINT_DIR, '<Name>StageModel.pth')`; file names unchanged | Same reason |
| 3 | `SEED = 69` with `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` | Issue 4. The split's own `random_state=69` is unchanged. cuDNN determinism was **not** forced (`cudnn.deterministic`), because it can slow training and changes nothing in the method: runs are repeatable on the same setup, not bit-exact |
| 38, 43, 50, 55 | `get_loss_fn(loss_type, weights)` → `get_loss_fn(loss_type, weights, device=device)` | Issue 9: the function's default `device="cuda"` ignored the notebook's `device`, so training stopped on a machine without CUDA. Every call has a `device` defined in an earlier cell (3, 29, 42, 49, 54). Found by the CPU smoke test, applied in a second pass after approval |
| all | Notebook metadata reduced to `kernelspec` and `language_info`. Every cell's metadata emptied (Kaggle `execution` timestamps, `trusted`, `collapsed`, Colab cell ids) | Request to remove Kaggle and Colab metadata. The Kaggle `dataSources` and the Colab GPU type were in the notebook-level metadata. No account data was present in this notebook (it was only in `evaluate.ipynb`) |

Not changed on purpose:
- the mixed Italian and English print messages;
- the typo "Grafichs saved";
- the duplicated `device = …` lines;
- the `!pip install -U` cell;
- the `validate` metric.

The Kaggle execution timestamps that documented the interrupted last run (see "Reconstructing the history of the code") were removed with the metadata. Their content is recorded above.

### Verification: CPU smoke test

There is no GPU on the preparation machine, so the corrected notebook was executed **cell by cell, in order, on CPU**, with this environment:
- Python 3.14.3;
- torch 2.14.1+cpu, torchvision 0.29.1;
- segmentation-models-pytorch 0.5.0, timm 1.0.30;
- albumentations 2.0.8, opencv 5.0.

The data were the 743 delivered samples. The harness reads the cells from the notebook file and applies these reductions **from outside**, without editing the notebook:
- after the split cell: 8 training, 24 validation and 4 test samples, `num_workers=0`;
- `n_epochs = 1` in every phase;
- `plt.show` closes the figures, as Colab's inline backend does;

Result, in two passes:
- **First pass, before issue 9 was fixed.** The run stopped at cell 38 with `Torch not compiled with CUDA enabled`; that is how issue 9 was found. To check the rest of the notebook, the harness passed the notebook's `device` to `get_loss_fn` from outside, and every code cell then ran.
- **Second pass, after the fix.** The same harness was run again **without** any workaround on `get_loss_fn`. Every code cell ran, including cell 38 where the first pass had stopped. All four phases trained and saved their checkpoints, the test and inference-time cells ran, and `evaluate.ipynb` loaded the new `FourthStageModel.pth` and scored 3 samples.

What the runs checked:

| Check | Outcome |
|---|---|
| Split on the 743 samples | 531 / 133 / 79, then reduced to the subset |
| Augmentation preview (cell 10) | read `DATA_ROOT/0022/rgb.jpg` |
| Phases 1–4 | each trained one epoch, saved its checkpoint (`First…FourthStageModel.pth` in `CHECKPOINT_DIR`) and wrote `training_curves_phase1…4.pdf` |
| Fine-tuning curve lists | `train_losses_fn` & co. defined, one entry each after phase 4 (no `NameError`) |
| Scheduler | `mode='max'`, stepped with the validation mIoU in every phase |
| Class weights | computed with `max_w=max_class_weight` (20) |
| Per-class legend | `smooth trail, traversable grass, rough trail, puddle, obstacle, non-traversable lw, high vegetation, sky`: 8 entries, no background |
| Test and inference-time cells | ran on the 4-image test subset |
| `evaluate.ipynb` | its code, with only the checkpoint path and `test_dir` substituted, loaded `FourthStageModel.pth` and scored 3 samples with the instructors' loop |

The scores of this run (mIoU around 0.06 after one epoch on 8 images) only show that the pipeline runs end to end. They say nothing about the model and are not reported anywhere.

### Negative control (incomplete)

The same harness was started on the **original** notebook. The only edits were path substitutions (Drive paths → local copy, `cleandataset/` → `train/`, Drive mount skipped), to show that it fails in phase 3. The system stopped the run for lack of memory during phase-1 training, after all cells up to 38 had run, so the `NameError` was **not** observed at run time.

In its place, a static check parsed every code cell (Python `ast`) and looked for reads of `train_losses_fn`, `val_losses_fn`, `val_ious_fn`, `val_class_ious_fn` before any assignment:
- original notebook: all four are read in cell 51 and never assigned;
- corrected notebook: no read before assignment.

The run-time negative control was not repeated.

---

## Figures in `assets/`

All figures were extracted from the report PDF with `pdfimages -png` and renamed. Code screenshots were not used.

| File | Report page | Content |
|---|--:|---|
| `discarded_sample_rgb.png`, `discarded_sample_mask.png` | 7 | sample removed during cleaning |
| `class_distribution_train.png` | 8 | pixel share per class, training split |
| `phase1_loss.png`, `phase1_val_miou.png`, `phase1_val_class_iou.png` | 15–16 | phase 1, 30 epochs |
| `phase2_loss.png`, `phase2_val_miou.png`, `phase2_val_class_iou.png` | 17 | phase 2, 48 epochs |
| `phase3_val_class_iou.png`, `phase3_val_miou.png`, `phase3_loss.png` | 19 | phase 3, 20 epochs |
| `phase4_loss.png` | 20 | phase 4, 30 epochs |
| `phase4_val_class_iou.png`, `phase4_final_epoch_output.png`, `phase4_val_miou.png` | 21 | phase 4 and the final-epoch printed output |

Each phase was matched to its figures by the number of epochs on the x axis and by the loss range (focal loss ≈ 0.13–0.45, CE + Dice ≈ 0.48–0.64). The figures with scene images show YCOR data (CC BY 4.0, attributed in the README).

---

## Personal data

- The authors appear by name only, with their consent.
- Student IDs and e-mail addresses (member list, report title page, last slide) are not published.
- The notebooks contain no names, IDs, e-mails or local paths. The only paths are Google Drive paths under `/content/drive/MyDrive/ML_Project_Work/`, which identify no one.
