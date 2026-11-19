"""Front end: SME loan portfolio selection (XGBoost + NSGA-II + investor steering).
Run:  streamlit run app.py
Needs sme_loan_ea.py and loan_data.py in the same folder."""
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymoo.optimize import minimize

from loan_data import generate
from sme_loan_ea import (Evaluator, FeasibleSampling, HVCallback, PortfolioProblem, SeededSampling, exact_front,
                         fit_pd, front, greedy, make_algo, pick_by_cap, prepare, quiet, random_pick, outcome_stats, scenarios)

NAVY, TEAL, AMBER, GREY, LIGHT = "#0B1F3A", "#1AB5A5", "#F2A000", "#5B6475", "#C5CCD8"
H = 36  # months in which a charge-off counts as a default

st.set_page_config(page_title="SME Loan Portfolio Optimizer", page_icon=":bar_chart:", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
.hero {{background: {NAVY}; border-left: 8px solid {TEAL}; padding: 1.3rem 1.6rem; border-radius: 10px; margin-bottom: 1rem;}}
.hero h1 {{color: #fff; font-family: Georgia, serif; font-size: 1.9rem; margin: 0 0 .3rem 0;}}
.hero p {{color: #B8C7DB; margin: 0; font-size: 1rem;}}
.card {{background: #fff; border: 1px solid #DDE3EC; border-radius: 10px; padding: 1rem 1.1rem; height: 100%;}}
.card h4 {{margin: 0 0 .35rem 0; color: {TEAL};}}
.card p {{margin: 0; color: #33415C; font-size: .93rem;}}
div[data-testid="stMetric"] {{background: #fff; border: 1px solid #DDE3EC; border-radius: 10px; padding: .7rem .9rem;}}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>SME Loan Portfolio Optimizer</h1>
<p>Choose which business loans to fund within a budget. Machine learning estimates risk, evolution searches for the best trade-offs, and you steer the final choice.</p></div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- back-end helpers
@st.cache_resource(show_spinner=False)
def build_world(n_loans, seed):
    """Create the loan set, split by time, train the default-risk model on older loans, score newer loans."""
    raw = generate(n_loans, seed)
    raw.columns = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in raw.columns]
    d, feat, asof = prepare(raw, SimpleNamespace(horizon=H))
    train = d[d.appr <= pd.Timestamp("2016-12-31") - pd.DateOffset(months=H)]
    test = d[(d.appr >= "2017-01-01") & (d.appr <= "2017-12-31") & (d.appr + pd.DateOffset(months=H) <= asof)]
    pdv, info = fit_pd(train, test, feat, seed)
    co = train[train.y == 1]
    lgd = float(np.clip((co["co_amt"] / co["amt"]).clip(0, 1).mean(), 0.1, 1.0))
    info.update(train_loans=len(train), test_loans=len(test), train_default_rate=float(train.y.mean()))
    return {"test": test.reset_index(drop=True), "pd": pdv, "lgd": lgd, "info": info}


class UICallback(HVCallback):
    """Tracks hypervolume each generation and moves the progress bar."""
    def __init__(self, ref, total, bar):
        super().__init__(ref)
        self.total, self.bar = total, bar

    def notify(self, algorithm):
        super().notify(algorithm)
        g = len(self.data["hv"])
        self.bar.progress(min(g / self.total, 1.0), text=f"Evolving portfolios: generation {g} of {self.total}")


def optimize(w, P, bar):
    t0 = time.time()
    rng = np.random.default_rng(P["seed"])
    test = w["test"]
    sel = rng.choice(len(test), min(P["pool"], len(test)), replace=False)
    pool, p = test.iloc[sel].reset_index(drop=True), w["pd"][sel]
    L, r, lgd = pool.amt.values, pool.rate.values, w["lgd"]
    net = L * (r * np.minimum(pool.term.values, H) / 12 * (1 - p) - lgd * p)
    exp_loss, budget = L * lgd * p, P["budget"] / 100 * L.sum()
    codes, uniq = pd.factorize(pool.sector)
    ev = Evaluator(L, net, scenarios(L, p, codes, len(uniq), lgd, P["rho"], P["scen"], rng), budget)
    ev2 = Evaluator(L, net, scenarios(L, p, codes, len(uniq), lgd, P["rho"], 5 * P["scen"], rng), budget)  # fresh, for reporting
    rand = np.array([random_pick(L, budget, rng) for _ in range(30)])
    ref = np.array([0.0, 1.1 * max(ev.F(rand)[:, 1].max(), ev2.F(rand)[:, 1].max())])
    xg = greedy(L, net, budget)
    bar.progress(0.0, text="Solving the exact baseline...")
    with quiet():
        xe = exact_front(L, net, exp_loss, budget, n_pts=25, time_limit=5)
    sampling = SeededSampling(L, net, exp_loss, budget) if P["seeded"] else FeasibleSampling(L, budget)
    cb = UICallback(ref, P["gens"], bar)
    res = minimize(PortfolioProblem(ev), make_algo(P["pop"], sampling, cb), ("n_gen", P["gens"]), seed=P["seed"], verbose=False, copy_algorithm=False)
    X, F1 = front(res, ev)
    return {"P": P, "pool": pool, "p": p, "ev": ev, "ev2": ev2, "X": X, "F1": F1, "F": ev2.F(X), "pop_X": res.pop.get("X"),
            "hv": cb.data["hv"], "xe": xe, "Fe": ev2.F(xe), "xg": xg, "Fg": ev2.F(xg)[0], "rand": rand, "Fr": ev2.F(rand),
            "budget": budget, "secs": time.time() - t0}


def steer(R, idx, bar):
    """Continue the search with R-NSGA-II, pulled toward the portfolio the investor chose."""
    P = R["P"]
    bar.progress(0.0, text="Steering the search toward your pick...")
    res = minimize(PortfolioProblem(R["ev"]), make_algo(P["pop"], R["pop_X"], ref=R["F1"][idx][None, :]),
                   ("n_gen", P["steer"]), seed=P["seed"], verbose=False, copy_algorithm=False)
    X, _ = front(res, R["ev"])
    return {"X": X, "F": R["ev2"].F(X)}


def describe(x, R):
    f = R["ev2"].F(x)[0]
    return {"Loans funded": int(np.sum(x)), "Capital used (%)": round(100 * R["pool"].amt.values[x].sum() / R["budget"], 1),
            "Expected net return (%)": round(-100 * f[0], 2), "Tail loss (%)": round(100 * f[1], 2), **{
                k: round(v, 2) for k, v in outcome_stats(x, R["pool"]).items() if k != "n_loans"}}


def front_chart(R, S=None, pick=None):
    F, Fe, Fr, Fg = R["F"], R["Fe"], R["Fr"], R["Fg"]
    o = np.argsort(Fe[:, 1])
    fig = go.Figure()
    fig.add_scatter(x=Fr[:, 1] * 100, y=-Fr[:, 0] * 100, mode="markers", name="Random picks", marker=dict(color=LIGHT, size=7))
    fig.add_scatter(x=Fe[o, 1] * 100, y=-Fe[o, 0] * 100, mode="lines+markers", name="Exact solver (linear version)",
                    line=dict(color="#2CA02C", dash="dash"), marker=dict(symbol="triangle-up", size=7))
    fig.add_scatter(x=F[:, 1] * 100, y=-F[:, 0] * 100, mode="markers", name="NSGA-II front", marker=dict(color="#1F77B4", size=8),
                    customdata=np.column_stack([R["X"].sum(1)]), hovertemplate="Tail loss %{x:.2f}%<br>Return %{y:.2f}%<br>%{customdata[0]:.0f} loans<extra>NSGA-II</extra>")
    if S is not None:
        fig.add_scatter(x=S["F"][:, 1] * 100, y=-S["F"][:, 0] * 100, mode="markers", name="After steering", marker=dict(color=AMBER, size=8))
    fig.add_scatter(x=[Fg[1] * 100], y=[-Fg[0] * 100], mode="markers", name="Greedy", marker=dict(symbol="star", color="#D62728", size=17))
    if pick is not None:
        fig.add_scatter(x=[pick[1] * 100], y=[-pick[0] * 100], mode="markers", name="Your pick",
                        marker=dict(symbol="circle-open", color=NAVY, size=20, line=dict(width=3)))
    fig.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tail loss (% of budget, lower is safer)",
                      yaxis_title="Expected net return (% of budget)", legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    n_loans = st.select_slider("Loans in the dataset", [20000, 30000, 40000, 60000], value=40000, key="n_loans")
    pool_n = st.slider("Loans in the candidate pool", 100, 500, 300, 50, key="pool")
    budget = st.slider("Budget (% of pool value)", 5, 50, 20, 5, key="budget")
    rho = st.slider("Sector correlation of defaults", 0.0, 0.6, 0.15, 0.05, key="rho",
                    help="Higher means defaults in the same industry cluster together, so tail losses are heavier.")
    pop = st.slider("Population size", 40, 200, 80, 20, key="pop")
    gens = st.slider("Generations", 20, 300, 80, 10, key="gens")
    steer_gens = st.slider("Steering generations", 10, 100, 30, 10, key="steer")
    seeded = st.checkbox("Greedy-seeded start (hybrid)", value=False, key="seeded")
    seed = st.number_input("Random seed", 0, 9999, 0, key="seed")
    run = st.button("Run optimization", type="primary", width="stretch", key="run")

if run:
    P = dict(pool=pool_n, budget=budget, rho=rho, pop=pop, gens=gens, steer=steer_gens, seeded=seeded, seed=int(seed), scen=1000)
    with st.spinner("Preparing loans and training the risk model (first run only)..."):
        W = build_world(n_loans, int(seed))
    bar = st.progress(0.0, text="Starting")
    st.session_state.R, st.session_state.W = optimize(W, P, bar), W
    st.session_state.pop("S", None)
    bar.empty()

R, W, S = st.session_state.get("R"), st.session_state.get("W"), st.session_state.get("S")
tab1, tab2, tab3, tab4 = st.tabs(["1. How it works", "2. Optimize", "3. Investor choice", "4. Results"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>1. Predict</h4><p>An XGBoost model estimates each loan\'s chance of default. It learns only from older loans, then scores newer loans it has never seen.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2. Evolve</h4><p>NSGA-II breeds loan portfolios. Each one is a yes/no choice per loan. It balances expected return against tail loss, within the budget.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3. Steer</h4><p>You see sample portfolios and pick the one closest to your taste. The search then focuses on that region of the trade-off.</p></div>', unsafe_allow_html=True)
    st.write("")
    if W is None:
        st.markdown("Set the options on the left and press **Run optimization**.")
    else:
        i = W["info"]
        st.subheader("Risk model on later loans")
        m = st.columns(5)
        m[0].metric("Training loans", f"{i['train_loans']:,}")
        m[1].metric("Later (test) loans", f"{i['test_loans']:,}")
        m[2].metric("AUC", f"{i['holdout_auc']:.3f}")
        m[3].metric("Mean predicted default", f"{100 * i['mean_predicted_pd']:.2f}%")
        m[4].metric("Actual default rate", f"{100 * i['actual_default_rate']:.2f}%")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    if R is None:
        st.info("Run the optimization first.")
    else:
        m = st.columns(4)
        m[0].metric("Candidate loans", len(R["pool"]))
        m[1].metric("Budget", f"${R['budget'] / 1e6:,.1f}M")
        m[2].metric("Portfolios on the front", len(R["X"]))
        m[3].metric("Run time", f"{R['secs']:.0f} s")
        fig = go.Figure(go.Scatter(y=R["hv"], mode="lines", line=dict(color=TEAL, width=3)))
        fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=30, b=10), title="Quality of the front by generation (hypervolume)",
                          xaxis_title="Generation", yaxis_title="Hypervolume")
        st.plotly_chart(fig, width="stretch")
        st.caption("A rising curve that flattens means the search has settled. If it is still climbing, add generations.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    if R is None:
        st.info("Run the optimization first.")
    else:
        F, X = R["F"], R["X"]
        left, right = st.columns([1.7, 1])
        with right:
            st.markdown("#### Step A: set a risk limit")
            lo, hi = float(np.round(100 * F[:, 1].min(), 2)), float(np.round(100 * F[:, 1].max(), 2))
            cap = st.slider("Largest tail loss you accept (% of budget)", lo, max(hi, lo + 0.01), float(np.round(np.median(100 * F[:, 1]), 2)), 0.01, key="cap")
            x_cap, f_cap = pick_by_cap(X, F, cap / 100)
            st.metric("Best expected return within your limit", f"{-100 * f_cap[0]:.2f}%", f"tail loss {100 * f_cap[1]:.2f}%", delta_color="off")
            st.markdown("#### Step B: steer the search")
            order = np.argsort(F[:, 1])
            idx = order[np.linspace(0, len(order) - 1, min(5, len(order))).astype(int)]
            labels = {int(i): f"{'ABCDE'[k]}: tail loss {100 * F[i, 1]:.2f}%, return {-100 * F[i, 0]:.2f}%, {int(X[i].sum())} loans" for k, i in enumerate(idx)}
            choice = st.radio("Which portfolio looks closest to what you want?", list(labels), format_func=labels.get, index=min(2, len(labels) - 1), key="choice")
            if st.button("Steer the search toward this choice", width="stretch", key="steer_btn"):
                bar = st.progress(0.0)
                st.session_state.S = steer(R, int(choice), bar)
                bar.empty()
                st.rerun()
            if S is not None:
                st.success(f"Steering done: {len(S['X'])} portfolios, now concentrated near your pick.")
        with left:
            pick = f_cap
            if S is not None:
                x_cap, f_cap = pick_by_cap(S["X"], S["F"], cap / 100)
                pick = f_cap
            st.plotly_chart(front_chart(R, S, pick), width="stretch")
        st.session_state.final = (x_cap, "NSGA-II + steering" if S is not None else "NSGA-II")

# ----------------------------------------------------------------------------- tab 4
with tab4:
    if R is None:
        st.info("Run the optimization first.")
    else:
        x_fin, name = st.session_state.get("final", (None, None))
        if x_fin is None:
            st.info("Open the Investor choice tab once to set your pick.")
        else:
            P = R["P"]
            x_ex, _ = pick_by_cap(R["xe"], R["Fe"], st.session_state.get("cap", 0) / 100)
            rows = {name + " (your pick)": describe(x_fin, R), "Exact solver, same risk limit": describe(x_ex, R), "Greedy (value per dollar)": describe(R["xg"], R)}
            rr = pd.DataFrame([outcome_stats(x, R["pool"]) for x in R["rand"]]).mean()
            fr = R["Fr"].mean(0)
            rows["Random picks (average of 30)"] = {"Loans funded": int(rr["n_loans"]), "Capital used (%)": np.nan, "Expected net return (%)": round(-100 * fr[0], 2),
                                                     "Tail loss (%)": round(100 * fr[1], 2), "return_pct": round(rr["return_pct"], 2),
                                                     "charged_off_pct": round(rr["charged_off_pct"], 2)}
            table = pd.DataFrame(rows).T.rename(columns={"return_pct": "What actually happened: return (%)", "charged_off_pct": "What actually happened: charged off (%)"})
            st.subheader("How the methods compare on later loans")
            st.dataframe(table, width="stretch")
            sel = R["pool"][x_fin].copy()
            sel["predicted_default_pct"] = np.round(100 * R["p"][x_fin], 2)
            sel["what_happened"] = np.where(sel["y"] == 1, "Charged off", "Repaid or active")
            out = sel[["sector", "amt", "rate", "term", "predicted_default_pct", "what_happened"]].rename(
                columns={"sector": "industry_code", "amt": "amount", "rate": "interest_rate", "term": "term_months"})
            c1, c2 = st.columns([1, 1.2])
            by = sel.groupby("sector")["amt"].sum().sort_values(ascending=False)
            bar_fig = go.Figure(go.Bar(x=by.index.astype(str), y=by.values / 1e6, marker_color=TEAL))
            bar_fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=30, b=10), title="Money funded by industry sector ($M)")
            c1.plotly_chart(bar_fig, width="stretch")
            c2.markdown("##### Loans in your portfolio")
            c2.dataframe(out.reset_index(drop=True), width="stretch", height=290, hide_index=True)
            st.download_button("Download portfolio as CSV", out.to_csv(index=False).encode(), "portfolio.csv", "text/csv", key="dl")
