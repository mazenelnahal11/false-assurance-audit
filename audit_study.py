import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


EXPECTED = {
    "MODE": "study", "STUDY_FROZEN": True,
    "STUDY_ARCHS": ["mobilenet_v3_large", "resnet34"], "STUDY_SEEDS": [11, 29, 47],
    "IG_STEPS": 256, "SPLIT_SEED": 1729, "CAL_FALSE_FLAG_RATE": 0.05,
    "MARGIN_DROP": 1.0, "PROTOCOL_VERSION": "0.3-study", "N_BOOT": 2000,
    "METHODS": ["gradcam", "ig", "neutralize"],
    "POLICIES": {"nominal_only": ["nominal", "dim", "bright"],
                 "context_diverse": ["nominal", "blur", "occlusion"]},
    "ORACLE_CONTEXTS": ["nominal", "blur", "occlusion"],
    "archive_sha256": "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5",
}


def assert_table(actual, recorded, keys, tolerance=1e-12):
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True),
        recorded[actual.columns].sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=0, atol=tolerance,
    )


def portfolio(raw, policies):
    raw = raw.assign(score=raw.score.astype("float32"))
    keys = ["image_id", "y", "arm", "method"]
    tables = []
    for name, views in policies.items():
        groups = raw.loc[raw.view.isin(views)].groupby(keys).score
        assert groups.size().eq(len(views)).all()
        tables.append(groups.max().rename("score").reset_index().assign(policy=name))
    return pd.concat(tables, ignore_index=True)


def risk_labels(raw, cutoff):
    risk = raw.aligned_correct & ~raw.swapped_correct & raw.margin_drop.ge(cutoff)
    outcomes = raw.assign(risky=risk)
    assert (outcomes.risky == raw.risky).all()
    keys = ["arm", "image_id"]
    stressed = outcomes.loc[outcomes.view.ne("nominal")].groupby(keys).risky.any()
    nominal = outcomes.loc[outcomes.view.eq("nominal")].set_index(keys).risky
    return pd.DataFrame({"risky": stressed, "nominal_risky": nominal}).reset_index()


def thresholds_from(cal, target):
    rows = []
    for (method, policy), items in cal.groupby(["method", "policy"]):
        values = np.sort(items.score.to_numpy())
        threshold = float(values[-(math.floor(len(values) * target) + 1)])
        rows.append(dict(method=method, policy=policy, threshold=threshold, calibration_n=len(values),
                         calibration_flag_rate=float(np.mean(values > threshold))))
    return pd.DataFrame(rows)


def review_at_quota(scores, risky, k):
    """Allocate exactly k slots, averaging uniformly across any cutoff ties."""
    cutoff = np.sort(scores)[-k]
    above, tied = scores > cutoff, scores == cutoff
    weight = (k - above.sum()) / tied.sum()
    caught = int((above & risky).sum()) + weight * (tied & risky).sum()
    return float(1 - caught / risky.sum())


def verify_study(source, output):
    def read(archive, name):
        return pd.read_csv(io.BytesIO(archive.read(name)))

    output.mkdir(parents=True, exist_ok=True)
    metrics, effects, quotas, quality, sensitivity, stress_only, class_metrics = ([] for _ in range(7))
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
        config = json.loads(archive.read("run_config.json"))
        for key, value in EXPECTED.items():
            assert config[key] == value, key
        planned = [(arch, seed) for arch in config["STUDY_ARCHS"] for seed in config["STUDY_SEEDS"]]
        gates = json.loads(archive.read("study_development_gates.json"))
        assert [(g["architecture"], g["seed"]) for g in gates] == planned
        assert all(g["all_passed"] and all(g["checks"].values()) for g in gates)
        checksums = json.loads(archive.read("checkpoint_manifest.json"))
        assert {x["path"] for x in checksums} == {
            f"{arch}_{seed}/{arm}.pt" for arch, seed in planned for arm in ("control", "shortcut")}
        assert all(len(x["sha256"]) == 64 and x["bytes"] > 0 for x in checksums)
        split = read(archive, "split_manifest.csv")
        assert split.image_id.is_unique and split.sha256.is_unique
        assert split.groupby("split").size().to_dict() == {"cal": 192, "dev": 144, "test": 448, "train": 1916}
        cases = {name: set(d.image_id) for name, d in split.groupby("split")}
        labels = split.set_index("image_id").y
        raw_keys = ["arm", "image_id", "view", "method"]
        for arch, seed in planned:
            tag = {"architecture": arch, "seed": seed}
            folder = f"{arch}_{seed}/"
            cal, test = (read(archive, folder + name) for name in ["cal_scores.csv", "view_scores.csv"])
            oracle = read(archive, folder + "oracle_cases.csv")
            assert set(cal.image_id) == cases["cal"] and set(cal.arm) == {"control"}
            assert set(test.image_id) == set(oracle.image_id) == cases["test"]
            assert len(test) == 2 * 448 * 5 * 3 and len(cal) == 192 * 5 * 3 and len(oracle) == 2 * 448 * 3
            assert set(test.arm) == set(oracle.arm) == {"control", "shortcut"}
            assert set(oracle.view) == set(config["ORACLE_CONTEXTS"])
            for frame in (test, cal):
                assert set(frame.method) == set(config["METHODS"])
                assert set(frame.view) == set().union(*config["POLICIES"].values())
                assert frame.score.notna().all() and not frame.duplicated(raw_keys).any()
            assert not oracle.duplicated(raw_keys[:-1]).any()
            for frame in (test, cal, oracle):
                assert np.array_equal(frame.y.to_numpy(), labels.loc[frame.image_id].to_numpy())
            risks = risk_labels(oracle, config["MARGIN_DROP"])
            cutoffs = thresholds_from(portfolio(cal, config["POLICIES"]), config["CAL_FALSE_FLAG_RATE"])
            assert_table(cutoffs, read(archive, folder + "thresholds.csv"), ["method", "policy"])
            d = portfolio(test, config["POLICIES"]).merge(cutoffs, on=["method", "policy"], validate="many_to_one")
            d = d.merge(risks, on=["arm", "image_id"], validate="many_to_one")
            d["flag"] = d.score.astype("float64") > d.threshold
            keys = ["arm", "method", "policy", "image_id"]
            assert_table(d[keys + ["risky", "nominal_risky", "flag"]], read(archive, folder + "decisions.csv"), keys)
            for (arm, method, policy), items in d.groupby(["arm", "method", "policy"]):
                pos = items.risky
                n_risky, n_missed = int(pos.sum()), int((pos & ~items.flag).sum())
                metrics.append(dict(arm=arm, method=method, policy=policy, n=len(items),
                                    n_risky=n_risky, n_missed=n_missed,
                                    missed_risk_rate=n_missed / n_risky if n_risky else np.nan,
                                    flag_rate=items.flag.mean(),
                                    oracle_negative_flag_rate=items.loc[~pos, "flag"].mean(), **tag))
            for method, items in d.loc[d.arm.eq("shortcut")].groupby("method"):
                pair = items.pivot(index=["image_id", "y", "risky"], columns="policy", values=["flag", "score"])
                index = pair.index.to_frame(index=False)
                risky = index.risky.to_numpy()
                nominal = pair[("flag", "nominal_only")].to_numpy(bool)
                diverse = pair[("flag", "context_diverse")].to_numpy(bool)
                delta = diverse.astype(int) - nominal.astype(int)
                rng = np.random.default_rng(seed + config["SPLIT_SEED"])
                strata = [np.flatnonzero(index.y.to_numpy() == y) for y in (0, 1)]
                boot = np.concatenate([rng.choice(s, (config["N_BOOT"], len(s))) for s in strata], axis=1)
                den = risky[boot].sum(1)
                assert (den > 0).all()
                lo, hi = np.quantile((delta[boot] * risky[boot]).sum(1) / den, [0.025, 0.975])
                effects.append(dict(method=method, n_risky=int(risky.sum()),
                                    diverse_only_flags_among_risky=int((risky & diverse & ~nominal).sum()),
                                    nominal_only_flags_among_risky=int((risky & nominal & ~diverse).sum()),
                                    delta_mrr=delta[risky].mean(), bootstrap_low=lo, bootstrap_high=hi, **tag))
                for quota in (0.05, 0.10, 0.20, 0.50):
                    k = math.floor(quota * len(pair))
                    a, b = (review_at_quota(pair[("score", policy)].to_numpy(), risky, k)
                            for policy in ("nominal_only", "context_diverse"))
                    quotas.append(dict(method=method, target_review_fraction=quota, slots=k, n=len(pair),
                                       nominal_mrr=a, diverse_mrr=b, delta_mrr=a-b, **tag))
            q = test.loc[test.method.eq("ig")].groupby(["arm", "view"]).ig_residual.quantile(.9)
            quality.extend(dict(arm=arm, view=view, ig_p90=value, **tag) for (arm, view), value in q.items())
            for key, table in (("risk_sensitivity", sensitivity), ("stress_only_contrasts", stress_only),
                               ("class_metrics", class_metrics)):
                table.append(read(archive, folder + key + ".csv").assign(**tag))
            budget = read(archive, folder + "portfolio_budget.csv")
            expected_cost = {"gradcam": (3, 3), "ig": (774, 768), "neutralize": (6, 0)}
            for row in budget.itertuples():
                fwd, back = expected_cost[row.method]
                assert (row.image_views, row.forward_images, row.backward_images) == (1344, 448*fwd, 448*back)
        verified_metrics, verified_effects = pd.DataFrame(metrics), pd.DataFrame(effects)
        assert_table(verified_metrics, read(archive, "study_metrics.csv"),
                     ["architecture", "seed", "arm", "method", "policy"])
        assert_table(verified_effects, read(archive, "study_paired_contrasts.csv"), ["architecture", "seed", "method"])
    for name, frame in {
        "verified_metrics": verified_metrics, "verified_paired_contrasts": verified_effects,
        "exploratory_fixed_review_quota": pd.DataFrame(quotas), "test_ig_residuals": pd.DataFrame(quality),
        "archived_risk_sensitivity": pd.concat(sensitivity), "archived_stress_only": pd.concat(stress_only),
        "archived_class_metrics": pd.concat(class_metrics),
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "verification.json").write_text(json.dumps({
        "archive": source.name, "sha256": digest, "checkpoint_pairs": len(planned),
        "test_images": 448, "calibration_images": 192,
        "verified": ["archive CRC and configuration", "split identities and labels", "all development gates",
                     "calibration thresholds", "oracle risk labels", "per-case flags", "audit budgets",
                     "primary metrics and paired bootstrap intervals"],
        "secondary_analysis": "Fixed review quotas are an exploratory analysis of the frozen test scores.",
    }, indent=2) + "\n", encoding="utf-8")
    print(verified_effects.to_string(index=False))
    print("Verified original decisions, metrics and intervals against the raw exported records.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify_study(args.archive, args.output)
