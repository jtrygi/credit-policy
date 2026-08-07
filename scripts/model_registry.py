"""Model versioning: save/load trained model artifacts, not just metrics.

Every model comparison so far (MODELING.md, FEATURE_SELECTION.md) recorded
scores but never persisted the fitted objects -- so nothing could actually
be reloaded to score a new applicant. This fixes that, with a convention
that keeps each saved version self-describing and reproducible.

Directory layout: models/<slug>/<version>/
  model.joblib        -- fitted sklearn-compatible estimator (LR, Tree, RF, MLP)
                          OR model.json for XGBoost (native format -- more
                          stable across xgboost version upgrades than pickling)
  preprocessing.joblib -- fitted StandardScaler (if any) + the exact list of
                          post-one-hot columns, in order. Required to score
                          new raw data consistently: categorical dummies
                          must be reindexed to this exact column set, or a
                          new applicant with a category the training data
                          didn't happen to see would silently misalign.
  metadata.json        -- features used, hyperparameters, git commit the
                          model was trained under, training timestamp, and
                          the val/test metrics achieved. A model file with
                          no metadata file next to it is not trustworthy --
                          treat metadata.json as mandatory, not optional.

Versions are never overwritten. Bump the version string (v1, v2, ... or a
date-based tag) for each new candidate; keep every version that was ever
seriously compared, not just the current winner, so FEATURE_SELECTION.md's
claims stay checkable against an actual artifact later.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import xgboost as xgb

MODELS_ROOT = Path("models")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return None


def save_model(slug, version, model, feature_cols, metrics, hyperparameters=None,
                scaler=None, notes=None):
    """Save a fitted model + its preprocessing + a metadata record.

    slug: short model family name, e.g. "xgboost-full-features"
    version: e.g. "v1" -- never overwrite an existing version
    model: the fitted estimator (sklearn-compatible, or an xgboost.XGBClassifier)
    feature_cols: exact post-one-hot column list/order the model expects
    metrics: dict of whatever's known (val_auc, test_auc, brier, ks, ...)
    hyperparameters: dict, for the record (esp. important for tuned models)
    scaler: fitted StandardScaler, if the model needs scaled inputs (LR, MLP)
    notes: free-text, e.g. "winner of the max-performance experiment"
    """
    out_dir = MODELS_ROOT / slug / version
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists -- versions are never overwritten. "
            f"Use a new version string."
        )
    out_dir.mkdir(parents=True)

    is_xgb = isinstance(model, xgb.XGBModel)
    if is_xgb:
        model.save_model(str(out_dir / "model.json"))
        model_file = "model.json"
        model_format = "xgboost_native"
    else:
        joblib.dump(model, out_dir / "model.joblib")
        model_file = "model.joblib"
        model_format = "joblib"

    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")

    metadata = dict(
        slug=slug,
        version=version,
        model_file=model_file,
        model_format=model_format,
        has_scaler=scaler is not None,
        feature_cols=list(feature_cols),
        n_features=len(feature_cols),
        metrics=metrics,
        hyperparameters=hyperparameters or {},
        git_commit=_git_commit(),
        saved_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"Saved {slug}/{version} -> {out_dir}")
    return out_dir


def load_model(slug, version):
    """Returns (model, scaler_or_None, metadata_dict)."""
    in_dir = MODELS_ROOT / slug / version
    with open(in_dir / "metadata.json") as f:
        metadata = json.load(f)

    if metadata["model_format"] == "xgboost_native":
        model = xgb.XGBClassifier()
        model.load_model(str(in_dir / metadata["model_file"]))
    else:
        model = joblib.load(in_dir / metadata["model_file"])

    scaler = joblib.load(in_dir / "scaler.joblib") if metadata["has_scaler"] else None
    return model, scaler, metadata


def list_versions(slug=None):
    """List all saved (slug, version) pairs, or versions for one slug."""
    if not MODELS_ROOT.exists():
        return []
    slugs = [slug] if slug else [p.name for p in MODELS_ROOT.iterdir() if p.is_dir()]
    out = []
    for s in slugs:
        slug_dir = MODELS_ROOT / s
        if not slug_dir.exists():
            continue
        for v in sorted(p.name for p in slug_dir.iterdir() if p.is_dir()):
            with open(slug_dir / v / "metadata.json") as f:
                meta = json.load(f)
            out.append((s, v, meta.get("metrics", {}), meta.get("saved_at")))
    return out


if __name__ == "__main__":
    for slug, version, metrics, saved_at in list_versions():
        print(f"{slug}/{version}  saved={saved_at}  metrics={metrics}")
