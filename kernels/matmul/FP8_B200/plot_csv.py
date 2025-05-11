import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('matmul_sm_sweep_results_min_time.csv')

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(df['SM_Count'], df['Time(us)'], label='Time(us)')
axes[0].set_xlabel('SM Count')
axes[0].set_ylabel('Time(us)')
axes[0].legend()
axes[1].plot(df['SM_Count'], df['TFLOPs'], label='TFLOPs')
axes[1].set_xlabel('SM Count')
axes[1].set_ylabel('TFLOPs')
axes[1].legend()
plt.savefig('matmul_sm_sweep_results.png', dpi=300)
