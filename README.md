# When Should Experts Expand?

COMPSCI 764 research project: a controlled study of novelty and gradient-interference triggers for expert expansion, with synthetic streams and a small MNIST neural extension.

## Status and findings

The final report and research artifact are ready for submission. On 24 October 2026, the authors confirmed completion of their assigned review, revisions and final manuscript reading. Reference formatting and the remapping accuracy rounding have been checked. AI assistance was used for implementation, analysis and draft preparation; both authors have reviewed the material and accept responsibility for the final submission. This public repository provides the code and recorded results for assessment; uploading the PDF to the course submission system remains a separate step.

Joint triggering reduces allocation relative to novelty-only on the tested MNIST streams, but fixed capacity matches its predictions. The retained-evidence variant provides no additional accuracy gain in these runs. Synthetic conflicting streams expose under-allocation. These results do not establish general superiority or a large-model continual-learning system.

## Completed author contributions

The following completed contributions were confirmed by the authors on 24 October 2026.

| Member | Completed contributions | Evidence to retain |
| --- | --- | --- |
| Mingqi Li | Reviewed the trigger and routing rules; verified code and data isolation; reproduced experiments; revised Methodology and Experimental Protocol | Test output, new run manifests, comparison of reproduced metrics, substantive manuscript revisions |
| Kaiyue Zhang | Read and verified cited sources; independently checked tables and statistical interpretation; analyzed failures and resource trade-offs; revised Related Work, Results, Discussion and Conclusion | Source notes, checked calculations, explanations of negative results, substantive manuscript revisions |
| Both | Refined research questions, Introduction and Abstract; cross-reviewed each other's work; completed final manuscript reading and verification | Mutual review notes and agreement on the final contribution statement |

For the presentation, Mingqi can explain the question, controller and experimental setup; Kaiyue can explain the evidence, limitations and conclusions. Both should be able to answer questions about the whole project. Retain records of the work and review performed under these responsibilities.

## Repository contents

- `experiments/`: unchanged experiment runners and 17 correctness tests.
- `results/`: six corrected canonical experiment sets, including per-run metrics, endpoint accuracy, window logs, paired comparisons, manifests and executed source snapshots. Synthetic: 640 runs including exploratory ablations. Neural: 50 initial runs plus 25 exploratory remapping runs.
- `05_神经网络补充实验方案.md` and `05b_标签重映射压力测试.md`: original protocol files, retained under their original names because the runners record their hashes.
- `data/mnist/provenance.json`: original data provenance and checksums.
- `Final_Report_Revised.pdf` and `sources/report/ieee/`: the revised report and its required LaTeX sources/figures.

Excluded: downloaded MNIST archives, model checkpoints and design arrays, obsolete/buggy runs, old report versions, slides, previews, build caches, course handouts and internal handoff notes. The original ZIP remains the full archival package. This repository supports rerunning experiments and inspecting recorded metrics; it does not include the saved weights required to independently re-evaluate the original trained models.

## Setup and tests

Use Python 3.12 (original runs used 3.12.14) and NumPy 2.3.5. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python -B -m unittest discover -s experiments -p 'test_experiment.py' -v
python -B -m unittest discover -s experiments/neural -p 'test_neural.py' -v
python scripts/check_results.py
```

On Windows, activate `.venv\Scripts\Activate.ps1` and set `$env:OPENBLAS_NUM_THREADS="1"` in PowerShell. Run commands from the repository root unless stated otherwise.

## Reproduce experiments

Always choose a new output directory: runners deliberately refuse to overwrite an existing run. Generated outputs below are ignored by Git.

```sh
python experiments/run_experiment.py --seeds 10 --output results/reproduction_synthetic
python experiments/neural/run_neural.py --prepare
python experiments/neural/run_neural.py --seeds 5 --output results/reproduction_neural
python experiments/neural/run_neural_stress.py --output results/reproduction_stress
```

`--prepare` downloads the four MNIST archives and verifies their MD5 checksums; it refreshes local provenance metadata. Neural runs create their own checkpoint and source-index arrays. Development/calibration use training positions 0–9999, formal streams use positions 10000–59999, and evaluation uses the official test split. The source contains no task identifiers passed into the learner. See the report and original protocols for settings and limitations.

Synthetic ablation settings are recorded in each canonical `manifest.json`; see `python experiments/run_experiment.py --help` for the persistence and threshold options. Timing varies by hardware, and numerical differences may occur across BLAS implementations.

## Read or build the report

Open [the revised report](Final_Report_Revised.pdf). With a LaTeX installation containing IEEEtran and the required packages:

```sh
cd sources/report/ieee
pdflatex main.tex
pdflatex main.tex
```

Figures and tables needed by the report are already included. The PDF and sources include revised bibliography formatting, the confirmed division of responsibilities and an AI-assistance statement. Both authors have confirmed completion of their final review.

## Result verification

`scripts/check_results.py` checks source hashes, run counts, matched-data hashes, and recomputes final accuracy and forgetting from endpoint CSVs. It also verifies reported means and sample standard deviations. This is a record-consistency check, not a fresh training run or proof of data authenticity. Checkpoint-level evaluation requires the omitted artifacts from the original package or a new training run.

## Final verification (24 October 2026)

- All 17 correctness tests passed (9 synthetic and 8 neural).
- The result checker passed for all 715 recorded runs across six canonical result sets.
- A fresh rerun of all 160 primary synthetic experiments exactly matched recorded final accuracy, forgetting, expert count, gradient-example count and retained-array bytes.
- The final report contains 10 body pages and one references page, with 11 references and an approximately 184-word abstract.
- Remapped Dual accuracy is consistently rounded to 91.78% (underlying mean: 91.775333...%).
- The fresh verification did not retrain the full neural experiments or re-evaluate omitted original checkpoints. Recorded-result consistency is distinct from independent neural reproduction.
