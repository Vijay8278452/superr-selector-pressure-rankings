import pandas as pd
import numpy as np
from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

def format_cell(col_name, val):
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(val)
    try:
        fval = float(val)
        if fval.is_integer() and any(k in col_name for k in ["Rank", "Balls", "Inns", "Spells", "Count", "Players", "Deliveries"]):
            return str(int(fval))
        if "SR" in col_name or "Econ" in col_name:
            return f"{fval:.1f}"
        if any(k in col_name for k in ["%", "Pct", "Prob", "Wkt", "Dism", "Dot", "Bound", "Extras"]):
            return f"{fval:.1f}\\%"
        if abs(fval) >= 100:
            return f"{fval:.1f}"
        return f"{fval:.3f}"
    except (ValueError, TypeError):
        pass
    s = str(val).replace("_", " ").replace("%", "\\%")
    return s

def clean_table(filename, caption, label, resize=True, col_align=None):
    csv_path = TABLES_DIR / f"{filename}.csv"
    tex_path = TABLES_DIR / f"{filename}.tex"
    df = pd.read_csv(csv_path)
    
    cols = df.columns.tolist()
    if col_align is None:
        col_align = "l" + "r" * (len(cols) - 1)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
    ]
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\toprule")
    headers = [str(c).replace("_", " ").replace("%", "\\%") for c in cols]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        row_vals = [format_cell(col, row[col]) for col in cols]
        lines.append(" & ".join(row_vals) + " \\\\")
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}")
    lines.append("\\end{table}")
    
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Cleaned {tex_path.name}")

if __name__ == "__main__":
    clean_table("table_1_data_audit", "Data verification audit and sample reconciliation", "tab:data_audit", resize=True, col_align="lrl")
    clean_table("table_2_model_performance", "Out of fold win probability and multiclass outcome model performance", "tab:model_performance", resize=True, col_align="lrrrr")
    clean_table("table_3_pressure_tiers", "Empirical match outcomes stratified by pressure tier", "tab:pressure_tiers", resize=True, col_align="lrrrrrrrr")
    clean_table("table_4_eligibility", "Player eligibility cut-offs and sample coverage under pressure", "tab:eligibility", resize=True, col_align="lrrrrrr")
    clean_table("table_5_top10_batters", "Top 10 batters under pressure with consensus rankings and metrics", "tab:top10_batters", resize=True)
    clean_table("table_6_top10_bowlers", "Top 10 bowlers under pressure with consensus rankings and metrics", "tab:top10_bowlers", resize=True)
    clean_table("table_7_ranks_11_to_20", "Players ranked 11 to 20 for batters and bowlers under pressure", "tab:ranks_11_to_20", resize=True, col_align="cllrcllrr")
    clean_table("table_8_sensitivity", "Sensitivity analysis across alternative thresholds and metric weights", "tab:sensitivity", resize=True, col_align="llcccc")
