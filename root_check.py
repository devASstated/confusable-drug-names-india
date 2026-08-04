import pandas as pd
d = pd.read_csv("results/collisions_molecule.csv")
tiny = d[d.brand_root.str.len() <= 2]
print(tiny.brand_root.nunique(), "roots,", len(tiny), "rows")
print(tiny[["brand_root", "composition", "example_name"]].to_string())
