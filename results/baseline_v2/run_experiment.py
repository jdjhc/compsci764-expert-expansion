"""Controlled CPU study of online expert expansion. Python 3.10+, NumPy only.
No external data, pretrained models, or task IDs enter the learner.
The context/predictor split is a deliberate toy assumption, not a visual benchmark.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

POLICIES = ("fixed", "novelty", "interference", "dual")
SCENARIOS = {
    "benign_shift": {"centres": [[0, 0], [5, 0], [0, 5]], "weights": [[1, .6], [1, .6], [1, .6]], "order": [0, 1, 2, 0, 1, 2]},
    "conflicting_shift": {"centres": [[0, 0], [5, 0], [0, 5]], "weights": [[1, .6], [-1, -.6], [.6, -1]], "order": [0, 1, 2, 0, 1, 2]},
    "mixed_recurrence": {"centres": [[0, 0], [5, 0], [0, 5], [0, 10]], "weights": [[1, .6], [1, .6], [-1, -.6], [-1, -.6]], "order": [0, 1, 2, 3, 0, 1, 2, 3]},
    "hidden_reversal": {"centres": [[0, 0], [0, 0]], "weights": [[1, .6], [-1, -.6]], "order": [0, 1, 0, 1, 0, 1]},
}

@dataclass
class Config:
    batch_size: int = 64
    replay_size: int = 256
    replay_batch: int = 32
    reference_batch: int = 64
    expert_cap: int = 4
    prototype_cap: int = 12
    persistence: int = 3
    gallery_persistence: int = 3
    cooldown: int = 3
    updates: int = 4
    windows_per_segment: int = 24
    warmup_windows: int = 24
    test_size: int = 1024
    learning_rate: float = .15
    replay_weight: float = .25
    l2: float = .0001
    context_sigma: float = .45
    label_noise: float = .05
    interference_threshold: float = .2
    gradient_floor: float = 1e-6
    novelty_threshold: float = 0.


def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -40., 40.)))


def loss_grad(w, x, y):
    z = x @ w[:2] + w[2]
    loss = np.mean(np.logaddexp(0., z) - y * z)
    residual = sigmoid(z) - y
    g = np.r_[x.T @ residual / len(x), np.mean(residual)]
    return float(loss), g


def conflict(a, b, floor):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < floor or nb < floor:
        return 0., False
    return float(max(0., -np.dot(a, b) / (na * nb))), True


def calibrate(cfg):
    """One independent, stationary calibration dataset. No evaluation labels used."""
    rng = np.random.default_rng(20261024)
    reference = rng.normal(0., cfg.context_sigma, (512, 2)).mean(axis=0)
    windows = rng.normal(0., cfg.context_sigma, (512, cfg.batch_size, 2)).mean(axis=1)
    distances = np.linalg.norm(windows - reference, axis=1)
    # Difference of two batch means is more variable than distance to 512 examples.
    independent_pairs = rng.normal(0., cfg.context_sigma, (512, 2, cfg.batch_size, 2)).mean(axis=2)
    pair_distances = np.linalg.norm(independent_pairs[:, 0] - independent_pairs[:, 1], axis=1)
    threshold = float(np.quantile(pair_distances, .995))
    return threshold, {"seed": 20261024, "quantile": .995, "sample_count": 512, "novelty_threshold": threshold, "reference_distance_q995": float(np.quantile(distances, .995)), "interference_threshold": cfg.interference_threshold, "interference_threshold_source": "prespecified, not fit to test data"}


def make_batch(rng, domain, scenario, n, cfg, id_start):
    x = rng.normal(size=(n, 2))
    c = rng.normal(scenario["centres"][domain], cfg.context_sigma, size=(n, 2))
    target = np.asarray(scenario["weights"][domain])
    y = ((x @ target) >= 0).astype(float)
    flips = rng.random(n) < cfg.label_noise
    y[flips] = 1 - y[flips]
    return {"x": x, "c": c, "y": y, "ids": np.arange(id_start, id_start+n, dtype=np.int64)}


def make_data(name, seed, cfg):
    s = SCENARIOS[name]
    train_rng = np.random.default_rng(seed + 10000)
    test_rng = np.random.default_rng(seed + 20000)
    stream = []
    cursor = 0
    for segment, domain in enumerate(s["order"]):
        for local in range(cfg.windows_per_segment):
            batch = make_batch(train_rng, domain, s, cfg.batch_size, cfg, cursor)
            batch.update(domain=domain, segment=segment, segment_end=local == cfg.windows_per_segment-1)
            stream.append(batch)
            cursor += cfg.batch_size
    tests = [make_batch(test_rng, d, s, cfg.test_size, cfg, -((d+1)*cfg.test_size)) for d in range(len(s["centres"]))]
    # Auxiliary reservoir priorities are common across every policy.
    priority_rng = np.random.default_rng(seed + 30000)
    priorities = [priority_rng.random(cfg.batch_size) for _ in stream]
    h = hashlib.sha256()
    for b in stream + tests:
        for k in ("x", "c", "y", "ids"):
            h.update(b[k].tobytes())
    return stream, tests, priorities, h.hexdigest()


class Memory:
    def __init__(self, cap):
        self.cap = cap
        self.x = np.empty((0, 2))
        self.c = np.empty((0, 2))
        self.y = np.empty(0)
        self.ids = np.empty(0, dtype=np.int64)
        self.priority = np.empty(0)

    def add(self, batch, priorities):
        if np.any(batch["ids"] < 0):
            raise AssertionError("Evaluation IDs must never enter replay memory")
        for k in ("x", "c", "y", "ids"):
            setattr(self, k, np.concatenate([getattr(self, k), batch[k]]))
        self.priority = np.r_[self.priority, priorities]
        if len(self.ids) > self.cap:
            chosen = np.argsort(self.priority, kind="stable")[-self.cap:]
            for k in ("x", "c", "y", "ids", "priority"):
                setattr(self, k, getattr(self, k)[chosen])
        assert len(self.ids) <= self.cap

    def nbytes(self):
        return sum(getattr(self, k).nbytes for k in ("x", "c", "y", "ids", "priority"))


class Learner:
    def __init__(self, policy, cfg, seed):
        self.policy, self.cfg = policy, cfg
        self.experts = [np.zeros(3)]
        self.anchors = []
        self.anchor_experts = []
        self.memory = Memory(cfg.replay_size)
        self.rng = np.random.default_rng(seed + 40000)
        self.pending = []
        self.streak = 0
        self.previous_candidate = None
        self.last_expansion = -100000
        self.expansion_events = 0
        self.blocked_events = 0
        self.gradient_calls = 0
        self.gradient_examples = 0
        self.maximum_memory_id = -1

    def route(self, contexts):
        if not self.anchors:
            return np.zeros(len(contexts), dtype=np.int64)
        distances = np.sum((contexts[:, None, :] - np.array(self.anchors)[None, :, :])**2, axis=2)
        return np.asarray(self.anchor_experts, dtype=np.int64)[np.argmin(distances, axis=1)]

    def gradient(self, w, x, y):
        self.gradient_calls += 1
        self.gradient_examples += len(x)
        return loss_grad(w, x, y)[1]

    def train_window(self, batch, priorities, t):
        cfg = self.cfg
        x, c, y, ids = (batch[k] for k in ("x", "c", "y", "ids"))
        if np.any(ids < 0):
            raise AssertionError("Evaluation data cannot enter training")
        if len(self.memory.ids):
            assert self.memory.ids.max() < ids.min(), "Only past samples may inform decisions"
            self.maximum_memory_id = max(self.maximum_memory_id, int(self.memory.ids.max()))
        mean_c = c.mean(axis=0)
        if not self.anchors:
            self.anchors.append(mean_c.copy())
            self.anchor_experts.append(0)
        distances = np.linalg.norm(np.asarray(self.anchors)-mean_c, axis=1)
        nearest = int(np.argmin(distances))
        novelty = float(distances[nearest])
        candidate = int(self.anchor_experts[nearest])
        memory_routes = self.route(self.memory.c)
        eligible = np.flatnonzero(memory_routes == candidate)
        interference, reliable = None, False
        if self.policy in ("interference", "dual") and len(eligible):
            # Deterministic prefix of reservoir entries, independent of evaluation.
            reference = eligible[:cfg.reference_batch]
            g_current = self.gradient(self.experts[candidate], x, y)
            g_old = self.gradient(self.experts[candidate], self.memory.x[reference], self.memory.y[reference])
            interference, reliable = conflict(g_current, g_old, cfg.gradient_floor)
        is_novel = novelty > cfg.novelty_threshold
        is_conflict = interference is not None and interference > cfg.interference_threshold
        condition = {"fixed": False, "novelty": is_novel, "interference": is_conflict, "dual": is_novel and is_conflict}[self.policy]
        if t < cfg.warmup_windows:
            condition = False
        if candidate != self.previous_candidate:
            self.streak = 0
        self.previous_candidate = candidate
        self.streak = self.streak+1 if condition else 0
        if is_novel:
            # Bounded transient buffer of means, not samples or labels.
            self.pending.append(mean_c.copy())
            self.pending = self.pending[-cfg.gallery_persistence:]
        else:
            self.pending = []
        expanded = blocked = False
        if self.streak >= cfg.persistence and t-self.last_expansion >= cfg.cooldown:
            if len(self.experts) < cfg.expert_cap and (not is_novel or len(self.anchors) < cfg.prototype_cap):
                self.experts.append(self.experts[candidate].copy())
                candidate = len(self.experts)-1
                expanded = True
                self.expansion_events += 1
                self.last_expansion = t
            else:
                blocked = True
                self.blocked_events += 1
            self.streak = 0
        # The SAME unsupervised gallery update applies to every policy.
        # New clusters attach to the expert selected after this window's decision.
        if expanded and is_novel:
            # A novel specialist needs its OWN context anchor immediately.
            # Reassigning the nearest old anchor here would corrupt old routing.
            self.anchors.append(np.mean(self.pending, axis=0))
            self.anchor_experts.append(candidate)
            self.pending = []
        elif is_novel and len(self.pending) >= cfg.gallery_persistence:
            if len(self.anchors) < cfg.prototype_cap:
                self.anchors.append(np.mean(self.pending, axis=0))
                self.anchor_experts.append(candidate)
            self.pending = []
        elif expanded:
            # Identical observable contexts cannot identify different label concepts.
            self.anchor_experts[nearest] = candidate
        elif not is_novel:
            self.anchors[nearest] = .95*self.anchors[nearest] + .05*mean_c
        # Recompute memory assignments after gallery admission/rewiring.
        eligible = np.flatnonzero(self.route(self.memory.c) == candidate)
        for _ in range(cfg.updates):
            g = self.gradient(self.experts[candidate], x, y)
            if len(eligible):
                replay_ids = self.rng.choice(eligible, cfg.replay_batch, replace=len(eligible)<cfg.replay_batch)
                old_g = self.gradient(self.experts[candidate], self.memory.x[replay_ids], self.memory.y[replay_ids])
                g = (1-cfg.replay_weight)*g + cfg.replay_weight*old_g
            g[:2] += cfg.l2*self.experts[candidate][:2]
            self.experts[candidate] -= cfg.learning_rate*g
        self.memory.add(batch, priorities)
        assert len(self.experts) <= cfg.expert_cap
        assert len(self.anchors) <= cfg.prototype_cap
        return {"window": t, "novelty": novelty, "interference": interference, "gradient_reliable": reliable, "condition": bool(condition), "expanded": expanded, "blocked": blocked, "experts": len(self.experts), "prototypes": len(self.anchors), "train_expert": candidate, "replay_count": len(self.memory.ids)}

    def evaluate(self, test):
        routes = self.route(test["c"])
        preds = np.zeros(len(test["y"]), dtype=bool)
        for e, w in enumerate(self.experts):
            chosen = routes == e
            preds[chosen] = (test["x"][chosen]@w[:2] + w[2]) >= 0
        return float(np.mean(preds == test["y"]))


def run_one(policy, name, seed, cfg, data):
    stream, tests, priorities, data_hash = data
    learner = Learner(policy, cfg, seed)
    traces, matrix = [], []
    seen = set()
    train_seconds = 0.
    for t, (batch, priority) in enumerate(zip(stream, priorities)):
        start = time.perf_counter()
        learner_batch = {k: batch[k] for k in ("x", "c", "y", "ids")}
        record = learner.train_window(learner_batch, priority, t)
        train_seconds += time.perf_counter()-start
        # Domain IDs below belong only to evaluator/logging, never to learner input.
        record.update(policy=policy, scenario=name, seed=seed, domain=batch["domain"], segment=batch["segment"])
        traces.append(record)
        seen.add(batch["domain"])
        if batch["segment_end"]:
            values = {d: learner.evaluate(tests[d]) for d in sorted(seen)}
            for d, accuracy in values.items():
                matrix.append({"policy": policy, "scenario": name, "seed": seed, "segment": batch["segment"], "test_domain": d, "accuracy": accuracy})
    final = {d: learner.evaluate(tests[d]) for d in sorted(seen)}
    forgetting = np.mean([max(r["accuracy"] for r in matrix if r["test_domain"] == d)-final[d] for d in sorted(seen)])
    # Array payloads only, not Python object/interpreter/RSS/peak training memory.
    expert_bytes = sum(w.nbytes for w in learner.experts)
    gallery_bytes = np.asarray(learner.anchors).nbytes + np.asarray(learner.anchor_experts, dtype=np.int64).nbytes
    pending_bytes = np.asarray(learner.pending).nbytes
    summary = {"scenario": name, "policy": policy, "seed": seed, "final_accuracy": float(np.mean(list(final.values()))), "forgetting": float(forgetting), "experts": len(learner.experts), "expansions": learner.expansion_events, "blocked_events": learner.blocked_events, "prototypes": len(learner.anchors), "gradient_calls": learner.gradient_calls, "gradient_examples": learner.gradient_examples, "train_seconds": train_seconds, "expert_parameters": len(learner.experts)*3, "expert_array_bytes": expert_bytes, "gallery_array_bytes": gallery_bytes, "replay_array_bytes": learner.memory.nbytes(), "persistent_array_bytes": expert_bytes+gallery_bytes+learner.memory.nbytes()+pending_bytes, "data_sha256": data_hash}
    return summary, traces, matrix


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(summaries):
    metrics = ("final_accuracy", "forgetting", "experts", "expansions", "blocked_events", "train_seconds", "gradient_examples", "persistent_array_bytes")
    rows = []
    for name in SCENARIOS:
        for policy in POLICIES:
            subset = [r for r in summaries if r["scenario"] == name and r["policy"] == policy]
            if not subset:
                continue
            result = {"scenario": name, "policy": policy, "n_seeds": len(subset)}
            for metric in metrics:
                values = [r[metric] for r in subset]
                result[metric+"_mean"] = float(np.mean(values))
                result[metric+"_sd"] = float(np.std(values, ddof=1)) if len(values)>1 else 0.
            rows.append(result)
    return rows


def paired_comparisons(summaries):
    rng = np.random.default_rng(20261025)
    rows = []
    for name in SCENARIOS:
        novel = {r["seed"]: r for r in summaries if r["scenario"] == name and r["policy"] == "novelty"}
        dual = {r["seed"]: r for r in summaries if r["scenario"] == name and r["policy"] == "dual"}
        seeds = sorted(set(novel)&set(dual))
        if not seeds:
            continue
        for metric in ("final_accuracy", "forgetting", "experts", "gradient_examples"):
            differences = np.array([dual[s][metric]-novel[s][metric] for s in seeds])
            boot = differences[rng.integers(0, len(seeds), size=(5000, len(seeds)))].mean(axis=1)
            rows.append({"scenario": name, "comparison": "dual_minus_novelty", "metric": metric, "n_pairs": len(seeds), "mean_difference": float(differences.mean()), "bootstrap_low95": float(np.quantile(boot, .025)), "bootstrap_high95": float(np.quantile(boot, .975))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"results"/"baseline_v1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--interference-threshold", type=float, default=.2)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--expert-cap", type=int, default=4)
    args = parser.parse_args()
    cfg = Config(interference_threshold=args.interference_threshold, persistence=args.persistence, expert_cap=args.expert_cap)
    if args.quick:
        cfg.windows_per_segment, cfg.warmup_windows, cfg.test_size = 8, 8, 256
    if args.seeds < 1 or args.persistence < 1 or args.expert_cap < 1:
        parser.error("seeds, persistence and expert cap must be positive")
    cfg.novelty_threshold, calibration = calibrate(cfg)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Choose a new directory to preserve raw runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"run_experiment.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"status": "running", "config": asdict(cfg), "calibration": calibration, "seeds": list(range(args.seeds)), "policies": POLICIES, "scenarios": SCENARIOS, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": "synthetic controlled experiment, not an image benchmark", "timing": "serial CPU learner updates including control, routing and replay; excludes evaluation and data generation", "memory_metric": "persistent NumPy array payloads only, not process RSS or peak memory"}
    manifest_path = args.output/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries, traces, matrices = [], [], []
    try:
        for name in SCENARIOS:
            for seed in range(args.seeds):
                data = make_data(name, seed, cfg)
                for policy in POLICIES:
                    summary, trace, matrix = run_one(policy, name, seed, cfg, data)
                    summaries.append(summary)
                    traces.extend(trace)
                    matrices.extend(matrix)
                print(f"{name}: seed {seed} completed", flush=True)
        for filename, rows in (("per_run.csv", summaries), ("window_log.csv", traces), ("accuracy_matrix.csv", matrices), ("summary.csv", aggregate(summaries)), ("paired_comparisons.csv", paired_comparisons(summaries))):
            write_csv(args.output/filename, rows)
        manifest["status"] = "completed"
        manifest["completed_runs"] = len(summaries)
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Results:", args.output, flush=True)

if __name__ == "__main__":
    main()
