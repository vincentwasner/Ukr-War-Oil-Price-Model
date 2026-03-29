import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path


print("Starting Brent oil price analysis...\n")

file_path = "data/brent_oil.csv"
date_path = "observation_date"
oil = "DCOILBRENTEU"
output_dir = "output/"


df = pd.read_csv(file_path)
#dataframe

print("Dataframe info:")


#df.info()

#df = df.rename(columns={"observation_date": "DATE"})
#did not update when print


df[date_path] = pd.to_datetime(df[date_path])
#df.info()

df[oil] = pd.to_numeric(df[oil])

#clean data
df = df.dropna(subset=[oil])

df = df.rename(columns={oil: "price"})
df = df.rename(columns={date_path: "date"})




#war info
ukr_war_date = pd.Timestamp("2022-02-24")
#iran_war_date = pd.Timestamp("2026.02.26")


df["post_war_ukr"] = (df["date"] >=ukr_war_date).astype(int)
#df["post_war_iran"] = (df["date"] >=iran_war_date).astype(int)


plt.figure(figsize=(12,6))

plt.plot(df["date"], df["price"], label = "Brent Oil Price", color = "blue")

plt.axvline(ukr_war_date, color = "red", label = "Ukraine War Start Date", linestyle = "--")
#plt.axvline(iran_war_date, color = "orange", label = "Iran War Start Date", linestyle = "--")

plt.xlabel("date")
plt.ylabel("price")

plt.legend()

plt.tight_layout()

#plt.show()

# dep var is price
y = df["price"]

#indep var is date
x_simple = df["post_war_ukr"]

x_simple = sm.add_constant(x_simple)

#price = b0 + (b1 * post_war_ukr)

model_simple = sm.OLS(y, x_simple).fit()


df = df.sort_values("date").copy()

df["time_index"] = range(len(df))

#prepare indep var for 2nd regression
#constant = price before war
#post_war_ukr = 0 if war before, 1 if after war
#time_index = indicates tiome passing for less ambiuguity

x_with_trend = df[["post_war_ukr","time_index"]]
x_with_trend = sm.add_constant(x_with_trend)

#fit OLS 2nd model
#checks whether there is a post war shift after accounting for the time trend

model_with_trend = sm.OLS(y, x_with_trend).fit()

print(model_with_trend.summary())

df["predicted"]=model_with_trend.predict(x_with_trend)

plt.figure(figsize= (12,6))
plt.plot(df["date"],df ["price"], label = "Actual Price")
post_war = df[df["date"] >= ukr_war_date]
plt.plot(post_war["date"], post_war["predicted"], label = "Predicted Price with Trend", color = "orange")

plt.axvline(ukr_war_date, color = "red", label = "Ukraine War Start Date", linestyle = "--")
plt.title("Actual vs. Predicted Oil Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.tight_layout()
plot_path= "output/actual_vs_predicted_prices.png"
plt.savefig(plot_path, dpi=150)
plt.close()