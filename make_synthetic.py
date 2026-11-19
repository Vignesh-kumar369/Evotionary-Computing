"""FAKE SBA-like loan data for demos and dry runs. Never report results from it.
Terminal:  python make_synthetic.py      (writes data_synth/synthetic_7a.csv)
Python:    from make_synthetic import generate; df = generate(40000)"""
import os
from loan_data import generate


if __name__ == "__main__":
    df = generate()
    os.makedirs("data_synth", exist_ok=True)
    df.to_csv("data_synth/synthetic_7a.csv", index=False)
    print("wrote data_synth/synthetic_7a.csv", df.shape, "charge-off share:", round((df.LoanStatus == "CHGOFF").mean(), 3))
