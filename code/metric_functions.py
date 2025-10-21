#%%
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

#%%
# --- Parameters ---
K0_range = np.linspace(0e-12, 1e-11, 10001)
# torsion parameter (set to 0 for Schwarzschild)

r_start = 1000  # starting radius (asymptotically flat region)
r_end = 1      # integrate inward toward smaller r

# --- Initial conditions ---
beta0 = 1e-6   # small deviation: e^β ≈ 1 + beta0
u0 = 1.0       # asymptotically u ~ 1

# Note here that e^alpha is the temporal term and e^beta is the radial term.

# Overall, I've noticed that at K0 = 0, the schwarzchild metric is obtained.
# e^alpha goes to -inf at r->0 and e^beta goes to inf at r->0.
# This is regular black hole behavior

# As K0 increases, the functions become softer
# At a critical point, e^alpha and e^beta both approach a finite number.
# e^beta->0 and e^alpha->a negative value. Is this an artifact of my ics?

# After this point, the r->0 behavor for the functions switches.
# e^alpha->inf at r->0 and e^beta->-inf at r->0
# The black hole works as a reflector??

# Ive found that the critical value of K is always 6 to the order of some magnitude depending on the initial beta.
# (while r_start = 100)
# K0_crit = 6e-9 with beta0 = 1e-5
# K0_crit = 6e-10 with beta0 = 1e-6
# K0_crit = 6e-11 with beta0 = 1e-7
# This might have something to do with the 6n^2f^2 factor in the equation for K

# Same thing with r_start
# (while beta0 = 1e-6)
# K0_crit = 6e-8 with r_start = 10
# K0_crit = 6e-10 with r_start = 100
# K0_crit = 6e-12 with r_start = 1000

#%%
def solve_metric_functions(K0):
  # --- System of ODEs ---
  def system(r, y):
    u, beta = y
    du_dr = -2 * u / r * (np.exp(beta) - 1)
    dbeta_dr = 0.5 * (r * u * K0 - 2 / r * (np.exp(beta) - 1))
    return [du_dr, dbeta_dr]

  # --- Integrate ---
  sol = solve_ivp(
    system, (r_start, r_end), [u0, beta0],
    dense_output=True, max_step=0.1, rtol=1e-9, atol=1e-9
  )

  # --- Extract results ---
  r_vals = sol.t
  u_vals, beta_vals = sol.y
  alpha_vals = beta_vals - np.log(u_vals)  # since u = e^{-α + β}

  return r_vals, beta_vals, alpha_vals

def find_horizon(r_vals, alpha_vals, threshold=1e-3):
    exp_alpha = np.exp(alpha_vals)
    mask = exp_alpha < threshold # wherever e^alpha ~ 0
    if np.any(mask):
        return r_vals[mask][0]
    else:
        return None

#%%

# --- Initial plot ---
fig, ax = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(bottom=0.25)

K0_init = K0_range[0]
r_vals, beta_vals, alpha_vals = solve_metric_functions(K0_init)
line_beta, = ax.plot(r_vals, np.exp(beta_vals), label=r"$e^{\beta(r)}$")
line_alpha, = ax.plot(r_vals, np.exp(alpha_vals), label=r"$e^{\alpha(r)}$")

r_h = find_horizon(r_vals, alpha_vals)
hline = None
if r_h:
  hline = ax.axvline(r_h, color='r', linestyle='--', label='Horizon')

ax.set_xlabel("r")
ax.set_ylabel("Metric functions")
ax.set_title(f"Torsionful metric solution (K₀={K0_init})")
ax.legend()
ax.grid(True)

# --- Slider ---
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "K₀", K0_range[0], K0_range[-1], valinit=K0_init, valstep=K0_range[1] - K0_range[0])

# --- Update function ---
def update(val):
  global hline
  K0 = slider.val
  r_vals, beta_vals, alpha_vals = solve_metric_functions(K0)
  line_beta.set_ydata(np.exp(beta_vals))
  line_alpha.set_ydata(np.exp(alpha_vals))
  
  if hline:
    hline.remove()
    hline = None
  
  r_h = find_horizon(r_vals, alpha_vals)
  if r_h:
    hline = ax.axvline(r_h, color='r', linestyle='--', label='Horizon')
    
  ax.set_title(f"Torsionful metric solution (K₀={K0})")
  fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
# %%
