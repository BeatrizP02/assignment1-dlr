import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("logs/2048_metrics.csv")

# Group and compute mean scores
summary = df.groupby(["algorithm", "persona"]).mean(numeric_only=True)[["score", "max_tile"]]
print("\nAverage Results:")
print(summary, "\n")

# Plot score progression
plt.figure(figsize=(8,5))
for (algo, persona), subdf in df.groupby(["algorithm", "persona"]):
    plt.plot(subdf["score"].values, label=f"{algo}-{persona}")

plt.title("2048 Agent Performance (PPO vs A2C, Maximizer vs Efficiency)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()

plt.savefig("plots/2048_agent_comparison.png")
plt.show()
