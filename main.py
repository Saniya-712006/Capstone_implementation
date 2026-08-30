"""
main.py

Single CLI entrypoint for the whole PhysChemCAL pipeline. Controls, via
flags (no code edits needed), exactly which of the two cost tiers you want:

  Basic pipeline (PhysChem + CAL train/eval only):
      python main.py --pkl-path /content/drive/MyDrive/capstone_data/drugood_lbap_ec50_scaffold_3d.pkl

  Same, but on a tiny truncated slice of the data for a fast pipeline check:
      python main.py --pkl-path <path> --smoke-test

  Full pipeline PLUS Phase-3 counterfactual explanation (expensive -- only
  runs when you explicitly ask for it):
      python main.py --pkl-path <path> --phase3

  Skip training entirely and just run Phase-3 against an already-trained
  checkpoint (useful for iterating on counterfactual generation without
  paying the training cost again):
      python main.py --pkl-path <path> --checkpoint checkpoints/best_model.pt --skip-train --phase3

  Resume an interrupted training run (e.g. after a Colab/Kaggle session
  timeout) from the most recent epoch instead of starting over, and push
  results/ to git every 10 epochs so progress survives future interruptions:
      python main.py --pkl-path <path> --checkpoint checkpoints/latest_model.pt --resume --push-every 10

See README.md for the full flag reference and Colab setup instructions.
"""

import argparse
import copy
import os
import random
from dataclasses import asdict

import torch

from configs.config import DEFAULT_CONFIG, Config
from src.data.dataset import get_dataloaders
from src.explain.counterfactual import generate_counterfactuals
from src.models.physchem_cal import PhysChemCAL
from src.training.evaluate import evaluate
from src.training.train import train
from src.utils.checkpoint import load_checkpoint
from src.utils.results_logger import ResultsLogger
from src.utils.seed import set_seed

SMOKE_TEST_DEFAULT_EPOCHS = 2
# Some DrugOOD lbap_ec50_scaffold molecules are large cyclic peptides (500+
# heavy atoms). The relational-force term's backward-pass memory cost scales
# with (atoms per batch)^2, so unless capped, a --smoke-test batch can land
# on a few huge peptides and OOM regardless of --batch-size. This default
# only applies to --smoke-test (and only when --max-atoms isn't given
# explicitly) -- real training runs are never size-filtered unless you ask.
SMOKE_TEST_DEFAULT_MAX_ATOMS = 150


def build_arg_parser() -> argparse.ArgumentParser:
    """Define every CLI flag this pipeline accepts, with the exact defaults pulled from configs/config.py so there is only ever one place hyperparameters are set."""
    p = argparse.ArgumentParser(
        description="PhysChemCAL: PhysChem-encoder + CAL-head training/eval, "
                     "with an optional Phase-3 counterfactual explainer.",
    )
    p.add_argument("--pkl-path", type=str, required=True,
                    help="Path to a cached, pre-3D-embedded DrugOOD .pkl file "
                         "(e.g. a mounted Google Drive path on Colab).")
    p.add_argument("--results-dir", type=str, default="results",
                    help="Folder for the per-day results/<date>.md log (default: results).")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                    help="Folder the best-validation-RMSE model is saved to during training.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to an existing checkpoint to load before training/eval "
                         "(e.g. checkpoints/best_model.pt). Combine with --skip-train to "
                         "skip straight to evaluation / --phase3.")
    p.add_argument("--skip-train", action="store_true",
                    help="Skip training entirely. Requires --checkpoint. Use this to run "
                         "--phase3 against an already-trained model without retraining.")
    p.add_argument("--resume", action="store_true",
                    help="Continue training from --checkpoint's saved epoch/optimizer/scheduler "
                         "state instead of starting fresh, up to --epochs total. Requires "
                         "--checkpoint (point it at checkpoint-dir/latest_model.pt, not "
                         "best_model.pt, so no completed epochs are lost). Contradicts --skip-train.")
    p.add_argument("--push-every", type=int, default=0,
                    help="Git-push the results/ folder every N epochs during training (0 = "
                         "disabled, the default). Best-effort -- failures are logged, never "
                         "fatal. Requires git identity + remote auth already configured before "
                         "training starts (see notebooks/run_colab.ipynb 'Authenticate to GitHub').")

    p.add_argument("--smoke-test", action="store_true",
                    help="Truncate every split to --smoke-n molecules and use a short "
                         f"epoch count (default {SMOKE_TEST_DEFAULT_EPOCHS} unless --epochs is "
                         "also given) -- for verifying the pipeline runs end-to-end, not for real results.")
    p.add_argument("--smoke-n", type=int, default=DEFAULT_CONFIG.SMOKE_N_PER_SPLIT,
                    help=f"Molecules per split under --smoke-test (default {DEFAULT_CONFIG.SMOKE_N_PER_SPLIT}).")
    p.add_argument("--max-atoms", type=int, default=None,
                    help="Skip molecules with more heavy atoms than this (some DrugOOD "
                         "entries are large peptides whose pairwise force term can exhaust "
                         "GPU memory regardless of --batch-size). Default: no cap for a real "
                         f"run; {SMOKE_TEST_DEFAULT_MAX_ATOMS} automatically under --smoke-test "
                         "unless you pass this explicitly. If you hit a CUDA OOM on a real "
                         "training run, this is the flag to reach for before shrinking --batch-size.")

    p.add_argument("--phase3", action="store_true",
                    help="Also run the Phase-3 CAL-guided counterfactual explainer after "
                         "training/loading the model. This is the expensive step -- off by default.")
    p.add_argument("--query-smiles", type=str, nargs="+", default=None,
                    help="Specific SMILES to explain under --phase3. If omitted, --n-queries "
                         "molecules are sampled from the OOD test split instead.")
    p.add_argument("--n-queries", type=int, default=3,
                    help="Number of OOD-test molecules to explain under --phase3 when "
                         "--query-smiles isn't given (default 3).")

    p.add_argument("--epochs", type=int, default=None,
                    help=f"Override training epochs (default {DEFAULT_CONFIG.EPOCHS}, or "
                         f"{SMOKE_TEST_DEFAULT_EPOCHS} under --smoke-test).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.BATCH_SIZE,
                    help=f"Training/eval batch size (default {DEFAULT_CONFIG.BATCH_SIZE}).")
    p.add_argument("--accumulation-steps", type=int, default=DEFAULT_CONFIG.ACCUMULATION_STEPS,
                    help="Gradient accumulation steps (effective batch = batch-size x this).")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.SEED,
                    help=f"Random seed (default {DEFAULT_CONFIG.SEED}).")
    p.add_argument("--device", type=str, default=None,
                    help="Torch device, e.g. 'cuda' or 'cpu' (default: cuda if available).")
    return p


def build_config(args: argparse.Namespace) -> Config:
    """Start from configs.config.DEFAULT_CONFIG and apply only the CLI overrides the user actually passed, so every other hyperparameter still comes from that single source of truth."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.BATCH_SIZE = args.batch_size
    config.ACCUMULATION_STEPS = args.accumulation_steps
    config.SEED = args.seed
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    elif args.smoke_test:
        config.EPOCHS = SMOKE_TEST_DEFAULT_EPOCHS
    return config


def build_model(config: Config, device: torch.device) -> PhysChemCAL:
    """Construct a PhysChemCAL model from `config`'s dimensions and move it to `device`."""
    model = PhysChemCAL(
        atom_ftr_dim=config.ATOM_FTR_DIM, bond_ftr_dim=config.BOND_FTR_DIM,
        hv_dim=config.HV_DIM, he_dim=config.HE_DIM, pq_dim=config.PQ_DIM,
        n_layer=config.N_LAYER, n_iteration=config.N_ITERATION, tau=config.TAU,
        rela_chunk=config.RELA_CHUNK, cal_dropout=config.CAL_DROPOUT,
    )
    return model.to(device)


def run_phase3(model: PhysChemCAL, data: dict, config: Config, device: torch.device,
               results_logger: ResultsLogger, query_smiles_list, n_queries: int) -> None:
    """Run the Phase-3 counterfactual explainer over either an explicit list of query SMILES, or `n_queries` molecules sampled from the OOD test split, logging each result."""
    if query_smiles_list is None:
        test_loader = data["test_loader"]
        pool = [s for batch in test_loader for s in batch["smiles"]]
        rng = random.Random(config.SEED)
        query_smiles_list = rng.sample(pool, min(n_queries, len(pool)))

    print(f"[phase3] explaining {len(query_smiles_list)} query molecule(s)...")
    for smi in query_smiles_list:
        result = generate_counterfactuals(
            model, smi, data["label_mean"], data["label_std"], device, config
        )
        cf_dicts = [
            {"smiles": cf.smiles, "pred": cf.pred, "similarity": cf.similarity, "direction": cf.direction}
            for cf in result["counterfactuals"]
        ]
        results_logger.log_counterfactuals(
            result["query_smiles"], result["query_pred"], result["causal_atom_count"],
            cf_dicts, alignment_score=result["alignment_score"],
        )
        print(f"[phase3] {smi}: pred={result['query_pred']:.4f}, "
              f"{len(result['counterfactuals'])} counterfactuals found, "
              f"alignment={result['alignment_score']}")


def main() -> None:
    """Parse CLI args and run: data setup -> (train unless --skip-train) -> test evaluation -> (Phase-3 counterfactuals if --phase3)."""
    args = build_arg_parser().parse_args()
    if args.skip_train and args.checkpoint is None:
        raise ValueError("--skip-train requires --checkpoint to load a model from.")
    if args.resume and args.checkpoint is None:
        raise ValueError("--resume requires --checkpoint (point it at checkpoint-dir/latest_model.pt).")
    if args.resume and args.skip_train:
        raise ValueError("--resume and --skip-train are contradictory: --resume continues training, "
                          "--skip-train skips it.")

    config = build_config(args)
    set_seed(config.SEED)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[main] device={device}, smoke_test={args.smoke_test}, phase3={args.phase3}")

    run_name = ("smoke_" if args.smoke_test else "") + ("train" if not args.skip_train else "eval")
    if args.phase3:
        run_name += "_phase3"
    results_logger = ResultsLogger(args.results_dir, run_name)
    results_logger.log_config({**vars(args), **asdict(config)})

    max_atoms = args.max_atoms
    if max_atoms is None and args.smoke_test:
        max_atoms = SMOKE_TEST_DEFAULT_MAX_ATOMS
    data = get_dataloaders(
        args.pkl_path, batch_size=config.BATCH_SIZE, smoke_test=args.smoke_test,
        smoke_n=args.smoke_n, max_atoms=max_atoms,
    )
    results_logger.log_note(
        f"Data: n_train={data['n_train']}, n_val={data['n_val']}, n_test={data['n_test']}, "
        f"label_mean={data['label_mean']:.4f}, label_std={data['label_std']:.4f}"
    )

    model = build_model(config, device)

    if args.checkpoint is not None:
        print(f"[main] loading checkpoint {args.checkpoint}")
        payload = load_checkpoint(args.checkpoint, model, map_location=str(device))
        # A loaded checkpoint's label stats are what the model was actually
        # trained against -- prefer them over freshly recomputed stats from
        # whatever .pkl happens to be passed this run.
        data["label_mean"] = payload["label_mean"]
        data["label_std"] = payload["label_std"]

    if not args.skip_train:
        resume_path = args.checkpoint if args.resume else None
        train(model, data, config, device, results_logger, args.checkpoint_dir,
              resume_path=resume_path, push_every=args.push_every)
        # reload best checkpoint before final test evaluation / phase3
        best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        payload = load_checkpoint(best_path, model, map_location=str(device))
        data["label_mean"] = payload["label_mean"]
        data["label_std"] = payload["label_std"]

    test_metrics = evaluate(model, data["test_loader"], device, data["label_mean"], data["label_std"])
    print(f"[main] OOD test RMSE: {test_metrics['rmse']:.4f} (n={test_metrics['n']})")
    results_logger.log_test_metrics(test_metrics)

    if args.phase3:
        run_phase3(model, data, config, device, results_logger, args.query_smiles, args.n_queries)


if __name__ == "__main__":
    main()
