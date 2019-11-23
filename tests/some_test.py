import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle(r"C:\Users\lukas\Desktop\log_res\01\01_25_k.pkl")

df_no_rand = df[df["greedy"] == 0.0]

fig, ax = plt.subplots(2)
ax[0].plot(df_no_rand["episode"], df_no_rand["final cash balance"])
ax[0].set_title("final cash balance")

ax[1].plot(df_no_rand["episode"], df_no_rand["deviation of x_1 from optimal x_1"])
ax[1].set_title("deviation of x_1 from optimal x_1")

fig.suptitle("Result after 25k without greedy choices episodes")
plt.show()


print(
    "Standard deviation {} and average {} final balance of the final amout of cash between 10-25k episode".format(
        np.std(df_no_rand["final cash balance"]),
        np.mean(df_no_rand["final cash balance"]),
    )
)
