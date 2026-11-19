"""Sample loan generator used by the app and the dry-run script."""
import numpy as np
import pandas as pd


def generate(N=80000, seed=0):
    rng = np.random.default_rng(seed)
    ASOF = pd.Timestamp("2025-03-31")
    appr = pd.Timestamp("2005-01-01") + pd.to_timedelta(rng.integers(0, 365 * 18, N), unit="D")
    amt = np.round(np.exp(rng.normal(11.8, 1.0, N)), -3).clip(10000, 5_000_000)
    term = rng.choice([60, 84, 120, 240, 300], N, p=[0.2, 0.2, 0.3, 0.1, 0.2])
    rate = np.round(rng.normal(0.075, 0.015, N).clip(0.03, 0.14), 4)
    naics = rng.choice([72, 44, 23, 54, 81, 62, 31, 48], N)
    new_biz = rng.random(N) < 0.3
    franchise = rng.random(N) < 0.1
    variable = rng.random(N) < 0.5
    sector_risk = {72: 0.6, 44: 0.2, 23: 0.5, 54: -0.4, 81: 0.1, 62: -0.5, 31: 0.0, 48: 0.3}
    shock = np.where((appr.year >= 2007) & (appr.year <= 2009), 0.8, 0.0)
    lin = (-4.1 + 1.6 * np.array([sector_risk[x] for x in naics]) + 40 * (rate - 0.075) + 1.1 * new_biz - 0.35 * franchise
           + 0.25 * variable - 0.12 * np.log(amt / 150000) + shock + rng.normal(0, 0.2, N))
    default = rng.random(N) < 1 / (1 + np.exp(-lin))
    co_date = appr + pd.to_timedelta(rng.integers(6, 60, N) * 30, unit="D")
    default &= co_date <= ASOF
    co_amt = np.where(default, amt * rng.uniform(0.3, 1.0, N), 0).round(0)
    paid_date = appr + pd.to_timedelta(rng.integers(12, 120, N) * 30, unit="D")
    paid = (~default) & (paid_date <= ASOF)
    status = np.where(default, "CHGOFF", np.where(paid, "PIF", "EXEMPT"))
    fmt = lambda x: pd.Series(x).dt.strftime("%m/%d/%Y")
    return pd.DataFrame({
        "AsOfDate": "03/31/2025", "Program": "7A", "GrossApproval": amt, "SBAGuaranteedApproval": (amt * 0.75).round(0),
        "ApprovalDate": fmt(appr), "TermInMonths": term, "NaicsCode": naics * 10000 + rng.integers(0, 9999, N),
        "InitialInterestRate": rate * 100, "LoanStatus": status,
        "PaidinFullDate": fmt(paid_date).where(paid, ""), "ChargeoffDate": fmt(co_date).where(default, ""),
        "GrossChargeoffAmount": co_amt, "BusinessType": rng.choice(["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"], N),
        "BusinessAge": np.where(new_biz, "New Business or 2 years or less", "Existing or more than 2 years old"),
        "FranchiseCode": np.where(franchise, "12345", ""), "SubProgram": rng.choice(["Guaranty", "Express"], N),
        "ProjectState": rng.choice(["CA", "TX", "NY", "FL", "OH", "IL"], N),
        "FixedorVariableInterestInd": np.where(variable, "V", "F")})
