import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('results/eda/figures', exist_ok=True)

df = pd.read_csv('results/eda/stats_por_tipo.csv')
df = df.sort_values('n_objetos', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(df['tipo'], df['n_objetos'], color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Número de objetos', fontsize=12)
ax.set_title('Distribución de supernovas por tipo — elasticc_1', fontsize=13)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

for bar, val in zip(bars, df['n_objetos']):
    ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/eda/figures/distribucion_por_tipo.png', dpi=150, bbox_inches='tight')
print('Figura guardada en results/eda/figures/distribucion_por_tipo.png')
