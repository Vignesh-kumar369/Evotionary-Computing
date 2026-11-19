#!/usr/bin/env python
"""SME loan portfolio selection.
Stage 1  XGBoost estimates each loan's probability of default (PD), trained only on older loans.
Stage 2  NSGA-II picks which loans to fund (binary chromosome) for two goals:
         maximize expected net return, minimize tail loss (CVaR with sector-correlated defaults).
Stage 3  Investor steering: show a few portfolios, take the pick, continue with R-NSGA-II.
Baselines: random, greedy (value per dollar), exact solver (MILP) on the linear version.
Evaluation: later loans the model never saw, with real outcomes."""
import argparse, glob, os, re, json, sys, contextlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.indicators.hv import HV


# ----------------------------------------------------------------------------- 1. data
CORE = ["approvaldate", "grossapproval", "initialinterestrate", "terminmonths",
        "naicscode", "loanstatus", "chargeoffdate", "grosschargeoffamount"]
CATS = {"businesstype": "btype", "businessage": "bage", "subprogram": "subprog",
        "projectstate": "state", "fixedorvariableinterestind": "fixvar"}


def load_raw(data_dir, pattern):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}/{pattern}. Put the SBA 7(a) FOIA csv files there.")
    frames = []
    for f in files:
        x = pd.read_csv(f, low_memory=False, encoding="latin-1")
        x.columns = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in x.columns]
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Columns found: {list(df.columns)}")
    print(f"Loaded {len(df):,} rows from {len(files)} file(s)")
    return df


def prepare(df, a):
    d = pd.DataFrame(index=df.index)
    d["appr"] = pd.to_datetime(df["approvaldate"], errors="coerce")
    d["amt"] = pd.to_numeric(df["grossapproval"], errors="coerce")
    d["rate"] = pd.to_numeric(df["initialinterestrate"], errors="coerce")
    d["term"] = pd.to_numeric(df["terminmonths"], errors="coerce")
    d["sector"] = df["naicscode"].astype(str).str.extract(r"(\d+)")[0].str[:2]
    d["status"] = df["loanstatus"].astype(str).str.upper().str.strip()
    d["co_date"] = pd.to_datetime(df["chargeoffdate"], errors="coerce")
    d["co_amt"] = pd.to_numeric(df["grosschargeoffamount"], errors="coerce").fillna(0)
    d["paid_date"] = pd.to_datetime(df["paidinfulldate"], errors="coerce") if "paidinfulldate" in df else pd.NaT
    if d["rate"].median() > 1:
        d["rate"] = d["rate"] / 100.0
    # keep loans that were funded and have a usable record (PIF, CHGOFF, EXEMPT = still active)
    d = d[~d["status"].isin(["CANCLD", "COMMIT", "NOT FUNDED"])]
    d = d.dropna(subset=["appr", "amt", "rate", "term", "sector"])
    d = d[(d.amt > 0) & (d.rate > 0) & (d.rate < 0.4) & (d.term > 0)].copy()
    d["log_amt"] = np.log(d["amt"])
    feat = ["log_amt", "rate", "term", "sector"]
    for raw, name in CATS.items():
        if raw in df.columns:
            d[name] = df.loc[d.index, raw].astype(str).str.strip()
            feat.append(name)
    if "sbaguaranteedapproval" in df.columns:
        d["guar"] = pd.to_numeric(df.loc[d.index, "sbaguaranteedapproval"], errors="coerce") / d["amt"]
        feat.append("guar")
    if "franchisecode" in df.columns:
        d["franchise"] = df.loc[d.index, "franchisecode"].notna().astype(int)
        feat.append("franchise")
    for c in feat:
        if not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype("category")
    # label: charged off within H months of approval (a fixed window, so every loan is judged equally)
    days = (d["co_date"] - d["appr"]).dt.days
    d["y"] = ((d["status"] == "CHGOFF") & (days <= a.horizon * 30.4375)).astype(int)
    # realized profit and loss over the same window (simple: no amortization, no SBA guarantee)
    yrs = np.minimum(d["term"], a.horizon) / 12.0
    paid_yrs = ((d["paid_date"] - d["appr"]).dt.days / 365.25).clip(lower=0)
    yrs = np.where(paid_yrs.notna(), np.minimum(yrs, paid_yrs), yrs)
    d["real_pnl"] = np.where(d["y"] == 1, -np.minimum(d["co_amt"], d["amt"]), d["amt"] * d["rate"] * yrs)
    asof = max(d["co_date"].max(), d["paid_date"].max() if "paidinfulldate" in df else d["appr"].max(), d["appr"].max())
    return d, feat, asof


# ----------------------------------------------------------------------------- 2. ML model
def fit_pd(train, test, feat, seed, diagnose=False):
    tr = train.sort_values("appr")
    cut = int(len(tr) * 0.8)  # newest 20% of the training period calibrates the probabilities
    fit, cal = tr.iloc[:cut], tr.iloc[cut:]
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", enable_categorical=True, eval_metric="auc", n_jobs=-1, random_state=seed)
    m.fit(fit[feat], fit["y"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=0.95)
    iso.fit(m.predict_proba(cal[feat])[:, 1], cal["y"])
    p = iso.predict(m.predict_proba(test[feat])[:, 1])
    if diagnose:  # a very high AUC can mean a feature leaks the outcome; this shows which features drive it
        print("\nTop features by importance:\n", pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False).head(8).round(3).to_string())
        for f in feat:
            cols = [c for c in feat if c != f]
            m2 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=seed)
            m2.fit(fit[cols], fit["y"])
            print(f"  holdout AUC without {f}: {roc_auc_score(test['y'], m2.predict_proba(test[cols])[:, 1]):.4f}")
        print()
    info = {"holdout_auc": float(roc_auc_score(test["y"], p)), "holdout_brier": float(brier_score_loss(test["y"], p)),
            "mean_predicted_pd": float(p.mean()), "actual_default_rate": float(test["y"].mean())}
    return p, info


# ----------------------------------------------------------------------------- 3. optimization problem
class Evaluator:
    """Objectives are divided by the budget so they read as fractions of capital."""
    def __init__(self, L, net, M, budget, alpha=0.95):
        self.L, self.net, self.M, self.budget = L, net, M, budget
        self.k = max(1, int(np.ceil((1 - alpha) * M.shape[0])))

    def F(self, X):
        X = np.atleast_2d(X).astype(float)
        f1 = -(X @ self.net) / self.budget                       # minimize negative net return
        loss = X @ self.M.T                                       # portfolio loss in each scenario
        tail = np.partition(loss, -self.k, axis=1)[:, -self.k:].mean(1) / self.budget  # CVaR
        return np.column_stack([f1, tail])

    def G(self, X):
        return (np.atleast_2d(X).astype(float) @ self.L - self.budget) / self.budget


class PortfolioProblem(Problem):
    def __init__(self, ev):
        super().__init__(n_var=len(ev.L), n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.ev = ev

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.ev.F(X)
        out["G"] = self.ev.G(X)[:, None]


class FeasibleSampling(Sampling):
    """Random portfolios that already fit the budget."""
    def __init__(self, L, budget):
        super().__init__()
        self.L, self.budget = L, budget

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            target, tot = self.budget * np.random.uniform(0.3, 1.0), 0.0
            for j in np.random.permutation(problem.n_var):
                if tot + self.L[j] <= target:
                    X[i, j] = True
                    tot += self.L[j]
        return X


class SeededSampling(Sampling):
    """Hybrid start: half the population comes from greedy rules that trade return against expected loss
    (a cheap heuristic, not the exact solver), the other half is random. Evolution then improves on them."""
    def __init__(self, L, net, exp_loss, budget):
        super().__init__()
        self.L, self.net, self.exp_loss, self.budget = L, net, exp_loss, budget

    def _do(self, problem, n_samples, **kwargs):
        X = FeasibleSampling(self.L, self.budget)._do(problem, n_samples)
        for k, lam in enumerate(np.linspace(0, 10, n_samples // 2)):
            score = (self.net - lam * self.exp_loss) / self.L
            x, tot = np.zeros(len(self.L), bool), 0.0
            for j in np.argsort(-score):
                if score[j] > 0 and tot + self.L[j] <= self.budget:
                    x[j] = True
                    tot += self.L[j]
            X[k] = x
        return X


class HVCallback(Callback):
    def __init__(self, ref):
        super().__init__()
        self.hv, self.data["hv"] = HV(ref_point=ref), []

    def notify(self, algorithm):
        self.data["hv"].append(float(self.hv(algorithm.opt.get("F"))))


def make_algo(pop, sampling, cb=None, ref=None):
    kw = dict(pop_size=pop, sampling=sampling, crossover=UniformCrossover(), mutation=BitflipMutation(),
              eliminate_duplicates=True, callback=cb)
    if ref is None:
        return NSGA2(**kw)
    return RNSGA2(ref_points=ref, epsilon=0.005, normalization="front", extreme_points_as_reference_points=False, **kw)


def front(res, ev):
    X = np.atleast_2d(res.X).astype(bool)
    return X, ev.F(X)


# ----------------------------------------------------------------------------- 4. baselines
@contextlib.contextmanager
def quiet():
    """Silences the C-level log lines the HiGHS solver prints."""
    sys.stdout.flush(); sys.stderr.flush()
    o1, o2, dn = os.dup(1), os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    try:
        yield
    finally:
        os.dup2(o1, 1); os.dup2(o2, 2)
        for f in (dn, o1, o2):
            os.close(f)


def scenarios(L, p, codes, n_sec, lgd, rho, S, rng):
    """One-factor model: defaults in the same industry sector move together."""
    Z, eps = rng.standard_normal((S, n_sec)), rng.standard_normal((S, len(L)))
    asset = np.sqrt(rho) * Z[:, codes] + np.sqrt(1 - rho) * eps
    return (asset < norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))[None, :]) * (L * lgd)[None, :]


def greedy(L, net, budget):
    x = np.zeros(len(L), bool)
    tot = 0.0
    for j in np.argsort(-net / L):
        if net[j] > 0 and tot + L[j] <= budget:
            x[j] = True
            tot += L[j]
    return x


def random_pick(L, budget, rng):
    x = np.zeros(len(L), bool)
    tot = 0.0
    for j in rng.permutation(len(L)):
        if tot + L[j] <= budget:
            x[j] = True
            tot += L[j]
    return x


def exact_front(L, net, exp_loss, budget, n_pts=40, time_limit=10):
    """Exact solver on the LINEAR version: maximize expected net return with a cap on expected loss
    (epsilon-constraint method). The result is then scored on the real tail-loss objective."""
    n = len(L)
    base = [LinearConstraint(L[None, :], -np.inf, budget)]
    kw = dict(integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    r0 = milp(-net, constraints=base, **kw)
    sols = [r0.x]
    for e in np.linspace(0.05, 1.0, n_pts) * float(exp_loss @ r0.x):
        r = milp(-net, constraints=base + [LinearConstraint(exp_loss[None, :], -np.inf, e)], **kw)
        if r.x is not None:
            sols.append(r.x)
    return np.unique(np.array(sols) > 0.5, axis=0)


# ----------------------------------------------------------------------------- 5. steering + evaluation
def investor_pick(F, lam, manual, ev, X, show=5):
    order = np.argsort(F[:, 1])
    idx = order[np.linspace(0, len(order) - 1, min(show, len(order))).astype(int)]
    if manual:
        print("\nSample portfolios from the current front:")
        for n, i in enumerate(idx, 1):
            print(f"  {n}. loans={int(X[i].sum())}  expected net return={-F[i,0]*100:.2f}%  tail loss={F[i,1]*100:.2f}% of budget")
        return idx[int(input("Pick one (number): ")) - 1]
    return idx[int(np.argmax(-F[idx, 0] - lam * F[idx, 1]))]  # simulated investor with risk aversion lam


def pick_by_cap(X, F, cap):
    ok = np.where(F[:, 1] <= cap)[0]
    i = ok[np.argmin(F[ok, 0])] if len(ok) else int(np.argmin(F[:, 1]))
    return X[i], F[i]


def realized(x, pool):
    s = pool[x]
    inv = s["amt"].sum()
    return {"n_loans": int(len(s)), "realized_return_pct": 100 * s["real_pnl"].sum() / inv if inv else np.nan,
            "charged_off_share_pct": 100 * s["y"].mean() if len(s) else np.nan}


def outcome_stats(x, pool):
    r = realized(x, pool)
    return {"n_loans": r["n_loans"], "return_pct": r["realized_return_pct"], "charged_off_pct": r["charged_off_share_pct"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--out", default="results"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=150); ap.add_argument("--steer-gens", type=int, default=50)
    ap.add_argument("--pool", type=int, default=500); ap.add_argument("--budget-frac", type=float, default=0.2)
    ap.add_argument("--scenarios", type=int, default=2000); ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--horizon", type=int, default=36, help="months in which a charge-off counts as default")
    ap.add_argument("--cutoff", default="2016-12-31"); ap.add_argument("--test-start", default="2017-01-01")
    ap.add_argument("--test-end", default="2017-12-31"); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pools", type=int, default=1, help="how many different random loan pools to test (more pools = more reliable realized results)")
    ap.add_argument("--diagnose", action="store_true", help="print feature importances and AUC with each feature removed")
    ap.add_argument("--seeded", action="store_true", help="start NSGA-II from greedy-rule portfolios plus random ones (hybrid)")
    ap.add_argument("--manual", action="store_true", help="you pick the portfolio yourself in the steering step")
    a = ap.parse_args(argv)
    tag = " (greedy-seeded)" if a.seeded else ""
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    d, feat, asof = prepare(load_raw(a.data_dir, a.pattern), a)
    cut = pd.Timestamp(a.cutoff)
    train = d[d.appr <= cut - pd.DateOffset(months=a.horizon)]          # labels fully known by the cutoff
    test = d[(d.appr >= a.test_start) & (d.appr <= a.test_end) & (d.appr + pd.DateOffset(months=a.horizon) <= asof)]
    print(f"Train loans {len(train):,} (default rate {train.y.mean():.3%}) | Test loans {len(test):,} (default rate {test.y.mean():.3%})")
    if len(train) < 5000 or len(test) < a.pool:
        raise SystemExit("Too few loans after the temporal split. Load more files or change the dates.")

    pdv, info = fit_pd(train, test, feat, a.seed, a.diagnose)
    print("PD model on later loans:", {k: round(v, 4) for k, v in info.items()})
    json.dump(info, open(os.path.join(a.out, "model_metrics.json"), "w"), indent=2)

    lgd = float(np.clip((train.loc[train.y == 1, "co_amt"] / train.loc[train.y == 1, "amt"]).clip(0, 1).mean(), 0.1, 1.0))
    rows, curves, plot_data = [], [], None
    for pi in range(a.pools):
        sel = rng.choice(len(test), a.pool, replace=False)
        pool, p = test.iloc[sel].reset_index(drop=True), pdv[sel]
        L, r, T = pool.amt.values, pool.rate.values, np.minimum(pool.term.values, a.horizon) / 12
        net = L * (r * T * (1 - p) - lgd * p)               # expected net return per loan
        exp_loss = L * lgd * p
        budget = a.budget_frac * L.sum()
        codes, uniq = pd.factorize(pool.sector)
        M = scenarios(L, p, codes, len(uniq), lgd, a.rho, a.scenarios, rng)               # scenarios the optimizer sees
        ev = Evaluator(L, net, M, budget)
        # fresh scenarios used only for reporting, so the optimizer cannot overfit its own risk estimate
        ev2 = Evaluator(L, net, scenarios(L, p, codes, len(uniq), lgd, a.rho, 5 * a.scenarios, rng), budget)
        print(f"LGD {lgd:.2f} | pool {a.pool} loans | budget {budget:,.0f} ({a.budget_frac:.0%} of pool value)")

        rand = [random_pick(L, budget, rng) for _ in range(30)]
        Xr = np.array(rand)
        ref = np.array([0.0, 1.1 * max(ev.F(Xr)[:, 1].max(), ev2.F(Xr)[:, 1].max())])   # hypervolume reference point
        hv = HV(ref_point=ref)
        xg = greedy(L, net, budget)
        with quiet():
            xe = exact_front(L, net, exp_loss, budget)
        Fe = ev2.F(xe)
        prob = PortfolioProblem(ev)

        for s in range(a.seeds):
            sd = a.seed + s
            r1 = minimize(prob, make_algo(a.pop, SeededSampling(L, net, exp_loss, budget) if a.seeded else FeasibleSampling(L, budget), HVCallback(ref)), ("n_gen", a.gens), seed=sd, verbose=False)
            X1, F1 = front(r1, ev)
            curves.append(r1.algorithm.callback.data["hv"])
            Xp = r1.pop.get("X")
            rA = minimize(prob, make_algo(a.pop, Xp), ("n_gen", a.steer_gens), seed=sd, verbose=False)
            XA, _ = front(rA, ev); FA = ev2.F(XA)                                             # plain NSGA-II, no steering
            i = investor_pick(F1, a.lam, a.manual, ev, X1)
            rB = minimize(prob, make_algo(a.pop, Xp, ref=F1[i][None, :]), ("n_gen", a.steer_gens), seed=sd, verbose=False)
            XB, _ = front(rB, ev); FB = ev2.F(XB)                                             # steered by the investor's pick
            cap = float(np.median(FA[:, 1]))                                   # investor's tail-loss tolerance
            util = lambda F: float(np.max(-F[:, 0] - a.lam * F[:, 1]))
            for name, (X, F) in {"NSGA-II" + tag: (XA, FA), "NSGA-II + steering" + tag: (XB, FB), "Exact solver (linear)": (xe, Fe)}.items():
                x, f = pick_by_cap(X, F, cap)
                rows.append({"pool": pi, "seed": sd, "method": name, "hypervolume": float(hv(F)), "exp_net_pct": -100 * f[0],
                             "tail_loss_pct": 100 * f[1], "investor_utility": util(F), **realized(x, pool)})
            for name, x in {"Greedy (value per dollar)": xg, "Random picks": rand[s % 30]}.items():
                f = ev2.F(x)[0]
                rows.append({"pool": pi, "seed": sd, "method": name, "hypervolume": np.nan, "exp_net_pct": -100 * f[0],
                             "tail_loss_pct": 100 * f[1], "investor_utility": np.nan, **realized(x, pool)})
            print(f"pool {pi} seed {sd}: front sizes plain {len(FA)}, steered {len(FB)}, exact {len(Fe)}")
            if s == 0 and pi == 0:
                plot_data = (FA, FB, Fe, ev2.F(xg)[0], ev2.F(Xr), ev2.F(X1[i])[0])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "per_seed_results.csv"), index=False)
    summ = res.groupby("method").agg(["mean", "std"]).drop(columns=["seed", "pool"], level=0).round(3)
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    cols = ["hypervolume", "exp_net_pct", "tail_loss_pct", "realized_return_pct", "charged_off_share_pct", "investor_utility"]
    print("\n=== mean over seeds (std in summary.csv) ===")
    print(res.groupby("method")[cols].mean().round(3).to_string())

    h = res.pivot_table(index=["pool", "seed"], columns="method", values="hypervolume")
    print(f"\nNSGA-II{tag} hypervolume beats the exact solver in {int((h['NSGA-II' + tag] > h['Exact solver (linear)']).sum())} of {len(h)} runs")
    FA, FB, Fe, Fg, Fr, Fpick = plot_data
    plt.figure(figsize=(7, 5))
    plt.scatter(Fr[:, 1] * 100, -Fr[:, 0] * 100, c="lightgray", label="Random picks")
    plt.scatter(FA[:, 1] * 100, -FA[:, 0] * 100, s=18, c="#1f77b4", label="NSGA-II front")
    plt.scatter(FB[:, 1] * 100, -FB[:, 0] * 100, s=18, c="#ff9f1c", label="After investor steering")
    o = np.argsort(Fe[:, 1])
    plt.plot(Fe[o, 1] * 100, -Fe[o, 0] * 100, "g^--", ms=5, label="Exact solver (linear version)")
    plt.scatter(Fg[1] * 100, -Fg[0] * 100, marker="*", s=160, c="red", label="Greedy")
    plt.scatter(Fpick[1] * 100, -Fpick[0] * 100, marker="o", s=90, facecolors="none", edgecolors="k", label="Investor pick")
    plt.xlabel("Tail loss (% of budget)"); plt.ylabel("Expected net return (% of budget)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "pareto_front.png"), dpi=150)
    plt.figure(figsize=(6, 4))
    for c in curves:
        plt.plot(c, alpha=0.4, c="#1f77b4")
    plt.plot(np.mean(curves, axis=0), c="k", label="mean of seeds")
    plt.xlabel("Generation"); plt.ylabel("Hypervolume"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "convergence.png"), dpi=150)
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()
