import numpy as np
import matplotlib.pyplot as plt
# importing libraries to solve vectors and draw images/pictures

#Given Inputs/parameters
Step_Size = [0.2, 0.05]
Eta_Inf = 8.0
# lets define some tolerance for the residual value 
Tol = 1e-8
# Intial guess vales for the f''(0)
X = (0.1,1.0)

# as the system is non linear now the gradient is written as Y'=F(Y). Now define right hand side function
def rhs(Y):
    y1, y2, y3 = Y
    return np.array([y2, y3, -0.5* y1 * y3])  # after reducing the 3rd order ODE to First Order ODE the new variables are replaced
# to get new variables after each iteration based on Eulers method New_value = Old_Value + Step_Size * rhs(Y)
def euler(Y,h):
    return Y + h * rhs(Y)
# Based on Runge Kutta Method the slopes at both ends and two arbitary mid vales are as defined
def rk4(Y,h):
    k1 = rhs(Y)
    k2 = rhs(Y + 0.5*h*k1)
    k3 = rhs(Y + 0.5*h*k2)
    k4 = rhs(Y + h*k3)
    return Y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)  # based on Simpsons rule 1-2-2-1

# Now store the functions with names in dictionary to use them flexibly
Iterations = {'Euler' : euler, 'RK4' : rk4}

# Now start integrationg Intial value problem with iteration method
def integrate(x, h, step):
    n = int(round(Eta_Inf / h))
    eta = np.linspace(0.0, n*h, n+1)
    Y = np.zeros((n + 1, 3))
    Y[0] = [0.0, 0.0, x]
    for i in range(n):
        Y[i+1] = step(Y[i], h)
    return eta, Y

# Define residual function to solve the equation for residual approaches to 0
def residual(x, h, step):
    eta_, Y = integrate(x, h, step)
    return Y[-1,1] - 1.0 # -1,1 defines rows and columns negative indicates tha first from the last row

# after iterationg and solving the residual at the both end points now iterate using shooting method 
def shoot(h, step, verbose=False):
    x0, x1 = X
    R0 = residual(x0, h, step)
    R1 = residual(x1, h, step)
    itr = [(x0, R0), (x1, R1)]
    for eta_ in range(50):
        x2 = x1 - R1 * (x1 - x0) / (R1 - R0)
        x0, R0 = x1, R1
        x1, R1 = x2, residual(x2, h, step)
        itr.append((x1, R1))
        if abs(R1) < Tol:
            break
    if verbose:
        for k, (x, R) in enumerate(itr):
            print(f"k = {k:2d}, x = {x:.8f}, Residual = {R:+.3e}")
    return x1

# to compare the results with high accurate solutions we need reference values
x_ref = shoot(0.001, rk4) # perform shooting with high accuracy step size
eta_ref, Y_ref = integrate(x_ref, 0.001, rk4) #based on the values of x_ref we integrate and get accurate solution

# lets store the results from both methods from the iteration to get gradient value at final end point at various eta guesses
results = {}
for name, step in Iterations.items():
    for h in Step_Size:
        print(f"\n {name}, d_eta = {h}:")
        x = shoot(h, step, verbose = True)
        eta, Y = integrate(x, h, step)
        results[(name, h)] = (x, eta, Y)

print("\n" + "-" * 70)
print(f"{'Iteration':8s} {'d_eta':>6s} {'f_pp(0)':>12s} {'error vs ref':>14s} "
      f"{'max|f_p - f_p,ref|':>20s}")
print("-" * 70)

for (name, h), (x, eta, Y) in results.items():
    Eta_Range = eta <= Eta_Inf + 1e-12 # define list of the range of eta (0,8.0) 
    fp_ref = np.interp(eta[Eta_Range], eta_ref, Y_ref[:, 1]) # gradient calculation at same eta with accuracy
    err_fp = np.max(np.abs(Y[Eta_Range, 1] - fp_ref))# absolute error betwee gradients at various eta
    print(f"{name:8s} {h:6.2f} {x:12.6f} {x - x_ref:+14.2e} {err_fp:20.2e}") 
print(f"{'Ref.RK4':8s} {0.001:6.3f} {x_ref:12.6f}")

 # print function values at the best solution for varios eta values from 0 to 8.0 with step size of 0.05 for RK4 method      
x, eta, Y = results[("RK4", 0.05)]
print("\nSolution table (RK4, d_eta = 0.05)")
print(f"{'eta':>5s} {'f':>10s} {'f_p = u*':>10s} {'f_pp':>10s} {'v*':>10s}")
for eta_value in range(0, 9):
    i = int(round(eta_value / 0.05))
    f, fp, fpp = Y[i]
    print(f"{eta_value:5.1f} {f:10.5f} {fp:10.5f} {fpp:10.5f} {0.5*(eta[i]*fp - f):10.5f}")

# Plot graphs for f, f', f'' for both methods and both step sizes
STYLE = {
    ("Euler", 0.2):  dict(color="tab:red",   ls="--", marker="o", ms=4, lw=1.3), # defining line style, line width and colour
    ("Euler", 0.05): dict(color="tab:orange", ls="--", lw=1.6),
    ("RK4", 0.2):    dict(color="tab:blue",  ls="-",  marker="s", ms=4, lw=1.3),
    ("RK4", 0.05):   dict(color="tab:green", ls="-",  lw=1.8),
}
quantities = [
    (0, "f",   r"$f(\eta)$",   "fig1_f.png"),
    (1, "f'",  r"$f'(\eta) = u/U_\infty$", "fig2_fp.png"),
    (2, "f''", r"$f''(\eta)$", "fig3_fpp.png"),
]
for col, title, ylabel, fname in quantities:
    plt.figure(figsize=(7, 5))
    for (name, h), (x, eta, Y) in results.items():
        Eta_Range = eta <= Eta_Inf + 1e-12
        plt.plot(eta[Eta_Range], Y[Eta_Range, col], label=fr"{name}, $\Delta\eta$ = {h}  ($f''(0)$ = {x:.4f})",
                         **STYLE[(name, h)])
    plt.xlabel(r"$\eta$")
    plt.ylabel(ylabel)
    plt.title(f"Blasius solution: {title}")
    plt.xlim(0, Eta_Inf)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=420)

# graphs for U* and V* for RK4 method with step size of 0.05
x, eta, Y = results[("RK4", 0.05)]
Eta_Range = eta <= Eta_Inf + 1e-12
f, fp = Y[Eta_Range, 0], Y[Eta_Range, 1]
u_star = fp
v_star = 0.5 * (eta[Eta_Range] * fp - f)
i99 = np.argmax(u_star >= 0.99)

plt.figure(figsize=(6, 6))
plt.plot(u_star, eta[Eta_Range], color="tab:green", lw=2)
plt.axhline(eta[Eta_Range][i99], color="gray", ls=":", lw=1)
plt.text(0.03, eta[Eta_Range][i99] + 0.12,
         fr"$u^* = 0.99$ at $\eta \approx$ {eta[Eta_Range][i99]:.2f}  ($\delta_{{99}}$)",
         fontsize=10)
plt.xlabel(r"$u^* = u/U_\infty = f'(\eta)$")
plt.ylabel(r"$\eta = y\sqrt{U_\infty/(\nu x)}$")
plt.title(r"Streamwise velocity $u^*$  (RK4, $\Delta\eta$ = 0.05)")
plt.xlim(0, 1.1)
plt.ylim(0, Eta_Inf)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_u_star.png", dpi=150)

plt.figure(figsize=(6, 6))
plt.plot(v_star, eta[Eta_Range], color="tab:purple", lw=2)
plt.axvline(v_star[-1], color="gray", ls=":", lw=1)
plt.text(v_star[-1] - 0.02, 0.3, fr"$v^*_\infty \approx$ {v_star[-1]:.4f}",
         ha="right", fontsize=10)
plt.xlabel(r"$v^* = \frac{v}{U_\infty}\sqrt{Re_x} = \frac{1}{2}(\eta f' - f)$")
plt.ylabel(r"$\eta = y\sqrt{U_\infty/(\nu x)}$")
plt.title(r"Wall-normal velocity $v^*$  (RK4, $\Delta\eta$ = 0.05)")
plt.xlim(0, 1.0)
plt.ylim(0, Eta_Inf)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig5_v_star.png", dpi=150)

print(f"\nf''(0) (RK4, d_eta = 0.05) = {x:.6f}")
print(f"boundary-layer thickness: u* = 0.99 at eta = {eta[Eta_Range][i99]:.2f}")
print(f"v* at eta = 8: {v_star[-1]:.5f}")

plt.show()
