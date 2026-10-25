from common import load, split, GROUPS, FEATURES

df = load()
train, test = split(df)
print("rows train/test:", len(train), len(test))
for name, d in [("train", train), ("test", test)]:
    print(name, "| subscribers:", int(d["y"].sum()), "| rate %:", round(d["y"].mean() * 100, 2),
          "| months:", d["month"].iloc[0], "->", d["month"].iloc[-1],
          "| euribor3m:", round(d["euribor3m"].iloc[0], 2), "->", round(d["euribor3m"].iloc[-1], 2))
print()
for g in GROUPS:
    t = test.groupby(g)["y"].agg(rows="size", subscribers="sum")
    t["judged"] = t["subscribers"] >= 30
    print(t.to_string(), "\n")
print(df[FEATURES].dtypes)
