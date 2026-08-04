import pandas as pd
d = pd.read_csv("results/collisions_molecule.csv")
d["len"] = d.brand_root.str.len()
print(d.groupby(d.len < 4).brand_root.nunique())   # how many short roots?
print(d[d.len < 4].brand_root.unique()[:40])
