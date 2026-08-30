"""
Plot the cyclic time-of-day feature f_t = [sin(2*pi*h/288), cos(2*pi*h/288)]
used in the observation, over one full 24-hour episode (288 steps),
to visually confirm the encoding behaves as described in the report.
"""
import numpy as np
import matplotlib.pyplot as plt

h = np.arange(288)
angle = 2 * np.pi * h / 288
sin_f = np.sin(angle)
cos_f = np.cos(angle)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: sin/cos over the course of one day (step index)
axes[0].plot(h, sin_f, label='sin(2*pi*h/288)', color='steelblue')
axes[0].plot(h, cos_f, label='cos(2*pi*h/288)', color='darkorange')
axes[0].set_xlabel('Step within day, h (5-min steps, 0-287)')
axes[0].set_ylabel('Feature value')
axes[0].set_title('Cyclic Time-of-Day Feature Over One Episode')
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(287, color='gray', linestyle=':', alpha=0.5)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: parametric plot (sin vs cos) -- should trace a perfect circle,
# confirming the encoding is injective (no two times map to the same point)
axes[1].plot(cos_f, sin_f, color='purple', linewidth=1.5)
axes[1].scatter([cos_f[0]], [sin_f[0]], color='green', s=80, zorder=5, label='h=0 (midnight)')
axes[1].scatter([cos_f[144]], [sin_f[144]], color='red', s=80, zorder=5, label='h=144 (noon)')
axes[1].set_xlabel('cos(2*pi*h/288)')
axes[1].set_ylabel('sin(2*pi*h/288)')
axes[1].set_title('Parametric View (confirms circular, injective encoding)')
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plotting/fig_time_feature_seasonality.png', dpi=150)
print("Saved: plotting/fig_time_feature_seasonality.png")

# Sanity checks
print(f"\nSanity checks:")
print(f"  h=0:   sin={sin_f[0]:.4f}, cos={cos_f[0]:.4f}")
print(f"  h=72:  sin={sin_f[72]:.4f}, cos={cos_f[72]:.4f}  (6:00 AM)")
print(f"  h=144: sin={sin_f[144]:.4f}, cos={cos_f[144]:.4f}  (Noon)")
print(f"  h=216: sin={sin_f[216]:.4f}, cos={cos_f[216]:.4f}  (6:00 PM)")
print(f"  h=287: sin={sin_f[287]:.4f}, cos={cos_f[287]:.4f}  (23:55 PM, near midnight)")
print(f"  Value range: sin=[{sin_f.min():.3f}, {sin_f.max():.3f}], cos=[{cos_f.min():.3f}, {cos_f.max():.3f}]")