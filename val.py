"""
CAPITA Evaluation Script (fixed)
Text metrics: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, SPICE  (all tasks)
VQA metrics : binary exact-match ACC (yes/no), open-ended token F1 (other QA),
composite VQA score = (ACC_bin + F1_open) / 2   (QA only, not caption)
"""

import json
import re
import string
import argparse
import traceback
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

SPICE_AVAILABLE = False
try:
    from pycocoevalcap.spice.spice import Spice
    SPICE_AVAILABLE = True
    print("✓ SPICE available")
except ImportError:
    print("⚠ SPICE not available — using fallback SPICE approximation")

from config import CAPITAConfig
from dataset import CAPITADataset, capita_collate_fn
from model import CAPITAModel


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

class VQAMetrics:

    def __init__(self):
        self.smoothing = SmoothingFunction().method4
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.stemmer = PorterStemmer()

        self.spice_scorer = None
        if SPICE_AVAILABLE:
            try:
                self.spice_scorer = Spice()
                print("✓ SPICE scorer initialized")
            except Exception as e:
                print(f"⚠️ SPICE initialization failed: {e}")

    def normalize_answer(self, answer: str) -> str:
        answer = str(answer).lower()
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", answer).strip()

    def compute_bleu(self, reference: str, candidate: str) -> dict:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if len(cand_tokens) == 0 or len(ref_tokens) == 0:
            return {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0}

        weights = {
            'BLEU-1': (1, 0, 0, 0),
            'BLEU-2': (0.5, 0.5, 0, 0),
            'BLEU-3': (1 / 3, 1 / 3, 1 / 3, 0),
            'BLEU-4': (0.25, 0.25, 0.25, 0.25),
        }
        out = {}
        for name, w in weights.items():
            n = sum(1 for x in w if x > 0)
            if len(cand_tokens) < n:
                out[name] = 0.0
                continue
            try:
                out[name] = float(sentence_bleu([ref_tokens], cand_tokens,
                                                weights=w, smoothing_function=self.smoothing))
            except Exception:
                out[name] = 0.0
        return out

    def compute_rouge(self, reference: str, candidate: str) -> dict:
        if len(candidate) == 0:
            return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
        s = self.rouge.score(reference, candidate)
        return {'ROUGE-1': s['rouge1'].fmeasure,
                'ROUGE-2': s['rouge2'].fmeasure,
                'ROUGE-L': s['rougeL'].fmeasure}

    def compute_meteor(self, reference: str, candidate: str) -> float:
        if len(candidate) == 0:
            return 0.0
        try:
            # BUG FIX: the import is named nltk_meteor, not meteor_score
            return float(nltk_meteor([reference.lower().split()], candidate.lower().split()))
        except Exception as e:
            print(f"⚠ METEOR failed: {e}")
            return 0.0

    def compute_spice_simple(self, reference: str, candidate: str) -> float:
        stop = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
                'had', 'do', 'does', 'did', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                'by', 'from', 'and', 'or', 'but', 'this', 'that', 'it', 'its', 'they',
                'them', 'their'}

        def toks(t):
            return {self.stemmer.stem(w.strip('.,!?')) for w in t.lower().split()
                    if w.strip('.,!?') not in stop and len(w) > 1}

        r, c = toks(reference), toks(candidate)
        if not r or not c:
            return 0.0
        inter = len(r & c)
        p, rec = inter / len(c), inter / len(r)
        return 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)

    def compute_spice(self, reference: str, candidate: str) -> float:
        if self.spice_scorer is None:
            return self.compute_spice_simple(reference, candidate)
        try:
            gts = {"0": [reference]}
            res = {"0": [candidate if candidate.strip() else "empty"]}
            avg_score, _ = self.spice_scorer.compute_score(gts, res)
            if isinstance(avg_score, dict):
                avg_score = avg_score.get('All', {}).get('f', 0.0)
            return float(avg_score)
        except Exception:
            return self.compute_spice_simple(reference, candidate)

    def compute_binary_accuracy(self, reference: str, candidate: str) -> float:
        return float(self.normalize_answer(reference) == self.normalize_answer(candidate))

    def compute_open_f1(self, reference: str, candidate: str) -> float:
        ref_tokens = self.normalize_answer(reference).split()
        cand_tokens = self.normalize_answer(candidate).split()
        if not ref_tokens and not cand_tokens:
            return 1.0
        if not ref_tokens or not cand_tokens:
            return 0.0
        overlap = sum((Counter(ref_tokens) & Counter(cand_tokens)).values())
        if overlap == 0:
            return 0.0
        p, r = overlap / len(cand_tokens), overlap / len(ref_tokens)
        return 2 * p * r / (p + r)

    def compute_all(self, reference: str, candidate: str, question: str = "",
                    question_type: Optional[str] = None) -> dict:
        """question_type: 'binary' | 'open' | 'caption' | None (auto-detect)."""
        reference = str(reference)
        candidate = str(candidate)

        metrics = {}
        metrics.update(self.compute_bleu(reference, candidate))
        metrics.update(self.compute_rouge(reference, candidate))
        metrics['METEOR'] = self.compute_meteor(reference, candidate)
        metrics['SPICE'] = self.compute_spice(reference, candidate)
        metrics['ACC'] = None
        metrics['F1_OPEN'] = None

        if question_type == 'caption':
            return metrics

        if question_type is None:
            is_binary = self.normalize_answer(reference) in {"yes", "no"}
        else:
            is_binary = question_type.lower() == "binary"

        if is_binary:
            metrics['ACC'] = self.compute_binary_accuracy(reference, candidate)
        else:
            metrics['F1_OPEN'] = self.compute_open_f1(reference, candidate)
        return metrics

    def compute_dataset_metrics(self, results: list) -> dict:
        bin_s = [r['ACC'] for r in results if r.get('ACC') is not None]
        open_s = [r['F1_OPEN'] for r in results if r.get('F1_OPEN') is not None]
        acc_bin = float(np.mean(bin_s)) if bin_s else 0.0
        f1_open = float(np.mean(open_s)) if open_s else 0.0
        return {
            'ACC_BIN': acc_bin,
            'F1_OPEN': f1_open,
            'VQA_SCORE': (acc_bin + f1_open) / 2,
            'N_BIN': len(bin_s),
            'N_OPEN': len(open_s),
        }


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

METRIC_NAMES = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
                'METEOR', 'SPICE', 'ACC', 'F1_OPEN']

# task name -> forced question type (None = auto-detect from reference)
TASK_QTYPE = {
    'caption': 'caption',
    'yes_no': 'binary',
    'uav_size': 'open',
    'environment': 'open',
    'motion_description': 'open',
}


def summarize(vals: list) -> dict:
    """Mean/std ignoring None (e.g. ACC is None for open questions)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}


def fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: CAPITAModel, loader: DataLoader, cfg: CAPITAConfig,
             metrics_calc: VQAMetrics, device: torch.device):
    model.eval()

    all_predictions: List[Dict] = []
    task_metrics = defaultdict(lambda: defaultdict(list))
    qa_results: List[Dict] = []          # QA only (no caption) -> dataset-level ACC/F1

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast(dtype=torch.float16, enabled=cfg.training.use_amp):
            preds = model(batch, generate=True)
            drone_feats = model.dual_stream_encoder(
                batch["frames"], batch["roi_patches"], batch["boxes"], batch["drone_mask"])
            swarm_feat = model.swarm_gnn(drone_feats, batch["boxes"], batch["drone_mask"])
            intent_repr = model.ctrm(swarm_feat)

        B = len(preds["video_id"])

        for b in range(B):
            vid_id = preds["video_id"][b]

            # ---- caption ----
            cap_pred = preds["caption_pred"][b]
            cap_gt = preds["caption_gt"][b]
            q = batch["caption_question"][b]
            m = metrics_calc.compute_all(cap_gt, cap_pred, question=q, question_type='caption')
            for k, v in m.items():
                task_metrics["all"][k].append(v)
                task_metrics["caption"][k].append(v)
            all_predictions.append({
                "video_id": vid_id, "task_type": "caption", "question": q,
                "generated_answer": cap_pred, "reference": cap_gt, **m,
            })

            # ---- other QA heads ----
            for q_key, a_key, task_name in [
                ("yes_no_question", "yes_no_answer_text", "yes_no"),
                ("size_question", "size_answer_text", "uav_size"),
                ("env_question", "env_answer_text", "environment"),
                ("motion_question", "motion_answer_text", "motion_description"),
            ]:
                q = batch[q_key][b]
                a = batch[a_key][b]
                if not q.strip():
                    continue

                qa_out = model.llm_head(intent_repr[b:b + 1], questions=[q], generate=True)
                pred_text = qa_out["generated_texts"][0]

                m = metrics_calc.compute_all(a, pred_text, question=q,
                                             question_type=TASK_QTYPE[task_name])
                for k, v in m.items():
                    task_metrics["all"][k].append(v)
                    task_metrics[task_name][k].append(v)
                qa_results.append(m)

                all_predictions.append({
                    "video_id": vid_id, "task_type": task_name, "question": q,
                    "generated_answer": pred_text, "reference": a, **m,
                })

    # Overall = "all" bucket only (previous code summed every bucket -> double counting)
    results_summary = {m: summarize(task_metrics["all"][m]) for m in METRIC_NAMES}
    per_task_summary = {
        t: {m: summarize(tm[m]) for m in METRIC_NAMES}
        for t, tm in task_metrics.items() if t != "all"
    }
    dataset_metrics = metrics_calc.compute_dataset_metrics(qa_results)

    return all_predictions, results_summary, per_task_summary, dataset_metrics


# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def save_results(all_predictions, results_summary, per_task_summary,
                 dataset_metrics, output_dir: Path, dataset_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions first
    pred_path = output_dir / 'predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)
    print(f"\n✓ Predictions saved to: {pred_path}")

    # metrics summary
    summary = {
        'model': 'CAPITA — Anti-UAV Intent Understanding',
        'dataset': dataset_name,
        'samples': len(all_predictions),
        'overall_metrics': results_summary,
        'per_task_metrics': per_task_summary,
        'vqa_metrics': dataset_metrics,
    }
    metrics_path = output_dir / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Metrics summary saved to: {metrics_path}")

    # paper table
    def pct(x):
        return round(x * 100, 2)

    table = {'Method': 'CAPITA', 'Dataset': dataset_name}
    for m in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1', 'ROUGE-2',
              'ROUGE-L', 'METEOR', 'SPICE']:
        table[m] = pct(results_summary[m]['mean'])
    table['ACC_BIN'] = pct(dataset_metrics['ACC_BIN'])
    table['F1_OPEN'] = pct(dataset_metrics['F1_OPEN'])
    table['VQA_SCORE'] = pct(dataset_metrics['VQA_SCORE'])
    table['ACC_YesNo'] = pct(per_task_summary.get('yes_no', {}).get('ACC', {}).get('mean', 0))
    for t, label in [('uav_size', 'F1_Size'), ('environment', 'F1_Environment'),
                     ('motion_description', 'F1_Motion')]:
        table[label] = pct(per_task_summary.get(t, {}).get('F1_OPEN', {}).get('mean', 0))

    table_path = output_dir / 'table_results.json'
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"✓ Table results saved to: {table_path}")

    # samples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    shown = set()
    for p in all_predictions:
        if p['task_type'] in shown:
            continue
        shown.add(p['task_type'])
        print(f"\nVideo: {p['video_id']} | Task: {p['task_type']}")
        print(f"Question:  {p['question']}")
        print(f"Generated: {p['generated_answer']}")
        print(f"Reference: {p['reference']}")
        print(f"BLEU-4: {fmt(p.get('BLEU-4'))} | METEOR: {fmt(p.get('METEOR'))} | "
              f"SPICE: {fmt(p.get('SPICE'))} | ACC: {fmt(p.get('ACC'))} | "
              f"F1: {fmt(p.get('F1_OPEN'))}")
        if len(shown) == 5:
            break

    # final tables
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE — CAPITA")
    print("=" * 70)
    print(f"\nBinary ACC  (N={dataset_metrics['N_BIN']}):  {dataset_metrics['ACC_BIN']:.4f}")
    print(f"Open F1     (N={dataset_metrics['N_OPEN']}):  {dataset_metrics['F1_OPEN']:.4f}")
    print(f"VQA score:  {dataset_metrics['VQA_SCORE']:.4f}")

    print(f"\n{'Metric':<12} {'Overall':>10} {'Caption':>10} {'Motion':>10}")
    print("-" * 46)
    for m in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'SPICE', 'ROUGE-L']:
        o = results_summary[m]['mean'] * 100
        c = per_task_summary.get('caption', {}).get(m, {}).get('mean', 0) * 100
        mo = per_task_summary.get('motion_description', {}).get(m, {}).get('mean', 0) * 100
        print(f"{m:<12} {o:>10.2f} {c:>10.2f} {mo:>10.2f}")

    print(f"\n{'Task':<22} {'ACC':>8} {'F1':>8} {'BLEU-4':>8} {'METEOR':>8}")
    print("-" * 58)
    for t, label in [('caption', 'Caption'), ('yes_no', 'Yes/No'),
                     ('motion_description', 'Motion'), ('uav_size', 'UAV Size'),
                     ('environment', 'Environment')]:
        s = per_task_summary.get(t, {})
        acc = s.get('ACC', {})
        f1 = s.get('F1_OPEN', {})
        acc_s = f"{acc['mean'] * 100:8.2f}" if acc.get('n') else "     n/a"
        f1_s = f"{f1['mean'] * 100:8.2f}" if f1.get('n') else "     n/a"
        b4 = s.get('BLEU-4', {}).get('mean', 0) * 100
        me = s.get('METEOR', {}).get('mean', 0) * 100
        print(f"{label:<22} {acc_s} {f1_s} {b4:>8.2f} {me:>8.2f}")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CAPITA Evaluation")
    parser.add_argument("--dataset", type=str, default="MultiUAV",
                        choices=["MultiUAV", "Anti-UAV", "NPS"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    args = parser.parse_args()

    cfg = CAPITAConfig()
    cfg.data.dataset_name = args.dataset
    cfg.training.batch_size = args.batch_size
    device = torch.device(cfg.training.device)

    output_dir = Path(args.output_dir) / args.dataset

    print("=" * 70)
    print(f"CAPITA Evaluation | Dataset: {args.dataset} | Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    eval_dataset = CAPITADataset(cfg, split=args.split)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True,
                             collate_fn=capita_collate_fn, drop_last=False)
    print(f"Evaluation samples: {len(eval_dataset)}")

    model = CAPITAModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Checkpoint loaded: {args.checkpoint}")

    metrics_calc = VQAMetrics()

    all_predictions, results_summary, per_task_summary, dataset_metrics = evaluate(
        model, eval_loader, cfg, metrics_calc, device)

    save_results(all_predictions, results_summary, per_task_summary,
                 dataset_metrics, output_dir, args.dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n❌ EVALUATION FAILED:")
        traceback.print_exc()
        raise
