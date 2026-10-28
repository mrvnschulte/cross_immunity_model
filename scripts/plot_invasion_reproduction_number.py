"""
Computes the invasion reproduction number and plots the invasion regions.
"""

from cycler import cycler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from cross_immunity_model import (
    critical_r01,
    critical_r02,
    critical_sigma,
    invasion_reproduction_number,
)


# Set parameters
dpi = 1200
fileformat = "pdf"
folder = "../plots"
alpha = 0.4
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


### STEP 1: Plot stability region and invasion reproduction number in
###         R01-R02-plane

r01 = np.linspace(1, 4, 201)
sigmas = np.linspace(0, 1, 5)

levels = np.linspace(0, 4, 17)
cmap = plt.get_cmap("viridis", len(levels) - 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

for sigma in sigmas:
    # Compute and plot invasion regions
    r02 = critical_r02(r01, sigma)
    plt.plot(r02, r01, "red")
    plt.plot(r01, r02, "red")
    plt.vlines(1, 0, 1, color="blue")
    plt.hlines(1, 0, 1, color="blue")
    plt.vlines(1, 1, r01[-1], color="purple")
    plt.hlines(1, 1, r01[-1], color="purple")
    plt.fill_between(
        r01,
        r01,
        r02,
        color="red",
        alpha=alpha,
        edgecolor="none",
        label="Disease free and single strain equilibria unstable",
    )
    plt.fill_betweenx(
        r01, r01, r02, color="red", alpha=alpha, edgecolor="none"
    )
    plt.fill_between(
        r01,
        r02,
        np.ones_like(r01),
        color="purple",
        alpha=alpha,
        edgecolor="none",
        label="Single strain equilibrium stable, but other strain could survive on its own",
    )
    plt.fill_betweenx(
        r01,
        r02,
        np.ones_like(r01),
        color="purple",
        alpha=alpha,
        edgecolor="none",
    )
    plt.fill_between(
        r01,
        1,
        0,
        color="blue",
        alpha=alpha,
        label="Single strain equilibrium stable and other strain could not survive on its own",
    )
    plt.fill_betweenx(r01, 1, 0, color="blue", alpha=alpha)
    plt.fill_between(
        [0, 1],
        0,
        1,
        color="black",
        alpha=alpha,
        label="Disease free equilibrium stable",
    )
    plt.xlim(0, r01[-1])
    plt.ylim(0, r01[-1])
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
    plt.ylabel(r"$R_{0,2}$", fontsize=fontsize_text)
    plt.tight_layout()
    plt.savefig(
        f"{folder}/stability_region_sigma={round(sigma,2)}.{fileformat}",
        dpi=dpi,
    )
    plt.show()
    plt.close()

    # Compute and plot invasion reproduction number
    invasion_r0 = np.zeros((len(r01), len(r01)))
    for i, r02 in enumerate(r01):
        invasion_r0[i] = invasion_reproduction_number(r02, r01, sigma)

    plt.contourf(
        r01,
        r01,
        invasion_r0.T,
        levels=levels,
        cmap=cmap,
        norm=norm,
    )
    plt.plot(r01, critical_r02(r01, sigma), color="tab:red")
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
    plt.ylabel(r"$R_{0,2}$", fontsize=fontsize_text)
    plt.tight_layout()
    plt.savefig(
        f"{folder}/invasion_reproduction_number_sigma={round(sigma,2)}.{fileformat}",
        dpi=dpi,
    )
    plt.show()
    plt.close()

# Plot legend of stability region plots as a separate plot
legend_handles = [
    Patch(
        facecolor="red",
        alpha=alpha,
        label="Disease free and single strain equilibria unstable",
    ),
    Patch(
        facecolor="purple",
        alpha=alpha,
        label=(
            "Single strain equilibrium stable,\nbut other strain "
            "could survive on its own"
        ),
    ),
    Patch(
        facecolor="blue",
        alpha=alpha,
        label=(
            "Single strain equilibrium stable\nand other strain "
            "could not survive on its own"
        ),
    ),
    Patch(
        facecolor="black",
        alpha=alpha,
        label="Disease free equilibrium stable",
    ),
]

legend_fig, legend_ax = plt.subplots()
legend_ax.axis("off")
legend_ax.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    fontsize=fontsize_text,
)
legend_fig.savefig(
    f"{folder}/stability_region_legend.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close(legend_fig)

# Plot colorbar of invasion reproduction number plots as separate plots
colorbar_fig, colorbar_ax = plt.subplots(figsize=(2, 5))
colorbar = colorbar_fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=colorbar_ax,
    boundaries=levels,
    spacing="uniform",
    orientation="vertical",
)
colorbar.ax.axhline(
    y=1,
    color="tab:red",
    linewidth=2,
    zorder=10,
)
colorbar.set_label(r"$R_{0,2 \to 1}$", fontsize=fontsize_text)
colorbar.ax.tick_params(labelsize=fontsize_ticks)
colorbar_fig.savefig(
    f"{folder}/invasion_reproduction_number_colorbar.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close(colorbar_fig)


### STEP 2: Plot invasion reproduction number dependence on single parameter

# Plot dependence on R01
r01s = np.linspace(1, 8, 201)
r02s = [1.5, 3]
sigmas = [0.4, 0.8]

fig = plt.figure()
plt.gca().set_prop_cycle(cycler(color=colors))
for r02 in r02s:
    for sigma in sigmas:
        invasion_r0 = [
            invasion_reproduction_number(r01, r02, sigma) for r01 in r01s
        ]
        plt.plot(
            r01s, invasion_r0, label=rf"$R_{{0,2}}={r02},\,\sigma={sigma}$"
        )

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
)
plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2\to1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.axhline(1, color="tab:red", linestyle="dashed")
plt.grid(alpha=alpha_grid)
plt.xlim(r01s[0], r01s[-1])
plt.ylim(0, r02s[-1])
plt.savefig(
    f"{folder}/invasion_reproduction_number_r01.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()

# Plot dependence on R02
r01s = [4, 8]
r02s = np.linspace(1, 4, 201)
sigmas = [0.4, 0.8]

fig = plt.figure()
plt.gca().set_prop_cycle(cycler(color=colors))
for r01 in r01s:
    for sigma in sigmas:
        invasion_r0 = [
            invasion_reproduction_number(r01, r02, sigma) for r02 in r02s
        ]
        plt.plot(
            r02s, invasion_r0, label=rf"$R_{{0,1}}={r01},\,\sigma={sigma}$"
        )

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
)
plt.xlabel(r"$R_{0,2}$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2\to1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.axhline(1, color="tab:red", linestyle="dashed")
plt.grid(alpha=alpha_grid)
plt.xlim(r02s[0], r02s[-1])
plt.ylim(0, r02s[-1])
plt.savefig(
    f"{folder}/invasion_reproduction_number_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()

# Plot dependence on sigma
r01s = [4, 8]
r02s = [1.5, 3]
sigmas = np.linspace(0, 1, 201)

fig = plt.figure()
plt.gca().set_prop_cycle(cycler(color=colors))
for r01 in r01s:
    for r02 in r02s:
        invasion_r0 = [
            invasion_reproduction_number(r01, r02, sigma) for sigma in sigmas
        ]
        plt.plot(
            sigmas,
            invasion_r0,
            label=rf"$R_{{0,1}}={r01},\,R_{{0,2}}={r02}$",
        )

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,1\to2}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.axhline(1, color="tab:red", linestyle="dashed")
plt.grid(alpha=alpha_grid)
plt.xlim(sigmas[0], sigmas[-1])
plt.ylim(0, r02s[-1])
plt.savefig(
    f"{folder}/invasion_reproduction_number_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 3: Plot bifurcation value of sigma

# Plot dependence on R01
r01s = np.linspace(1, 10, 1001)
r02s = [1.5, 3, 4.5]

fig = plt.figure()
for r02 in r02s:
    c_sigma = [critical_sigma(r01, r02) for r01 in r01s]
    plt.plot(r01s, c_sigma, label=rf"$R_{{0,2}}={r02}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
plt.ylabel(r"$\sigma_{2\to 1}$", fontsize=fontsize_text)
plt.xlim(r01s[0], r01s[-1])
plt.ylim(0, 1)
plt.grid(alpha=alpha_grid)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.savefig(
    f"{folder}/critical_sigma_r01.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
plt.close()

# Plot dependence on R02
r01s = [4, 6, 8]
r02s = np.linspace(1, 4, 201)

fig = plt.figure()
for r01 in r01s:
    c_sigma = [critical_sigma(r01, r02) for r02 in r02s]
    plt.plot(r02s, c_sigma, label=rf"$R_{{0,1}}={r01}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$R_{0,2}$", fontsize=fontsize_text)
plt.ylabel(r"$\sigma_{2\to 1}$", fontsize=fontsize_text)
plt.xlim(r02s[0], r02s[-1])
plt.ylim(0, 1)
plt.grid(alpha=alpha_grid)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.savefig(
    f"{folder}/critical_sigma_r02.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
plt.close()

# Plot contour plot
r01s = np.linspace(1, 4, 201)
critical_sigmas = np.zeros((len(r01s), len(r01s)))
for i, r01 in enumerate(r01s):
    critical_sigmas[i] = critical_sigma(r01, r01s)

fig = plt.figure()
levels = np.linspace(0, 1, 11)
cs = plt.contourf(r01s, r01s, critical_sigmas.T, levels=levels, vmin=0, vmax=1)
plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
# plt.title("Critical Sigma")
cbar = plt.colorbar(cs)
cbar.set_label(r"$\sigma_{2\to 1}$", fontsize=fontsize_text)
cbar.ax.tick_params(labelsize=fontsize_ticks)
plt.savefig(
    f"{folder}/critical_sigma.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
plt.close()


### STEP 4: Plot bifurcation value of R01

# Plot dependence on R02
sigmas = [0.4, 0.6, 0.8]
r02s = np.linspace(1, 10, 100001)

fig = plt.figure()
for sigma in sigmas:
    c_r01 = [critical_r01(r02, sigma) for r02 in r02s]
    plt.plot(r02s, c_r01, label=rf"$\sigma={sigma}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$R_{0,2}$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,1}^{2\to 1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.ylim(top=20, bottom=0)
plt.xlim(1, 5)
plt.savefig(
    f"{folder}/critical_r01_r02.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
plt.close()

# Plot dependence on sigma
sigmas = np.linspace(0, 1, 100001)
r02s = [1.5, 3, 4.5]

fig = plt.figure()
for r02 in r02s:
    c_r01 = [critical_r01(r02, sigma) for sigma in sigmas]
    plt.plot(sigmas, c_r01, label=rf"$R_{{0,2}}={r02}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,1}^{2\to 1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.ylim(top=20, bottom=0)
plt.xlim(0, 1)
plt.savefig(
    f"{folder}/critical_r01_sigma.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
plt.close()

# Contour plot
r02s = np.linspace(1, 4, 401)
sigmas = np.linspace(0, 1, 401)

critical_r01s = np.zeros((len(sigmas), len(r02s)))
for i, sigma in enumerate(sigmas):
    critical_r01s[i] = critical_r01(r02s, sigma)

levels = np.linspace(1.0, 20, 20)
cr = plt.contourf(sigmas, r02s, critical_r01s.T, levels=levels, extend="max")
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
cbar = plt.colorbar(cr)
cbar.set_label(r"$R_{0,1}^{2\to1}$", size=fontsize_text)
cbar.ax.tick_params(labelsize=fontsize_ticks)
plt.savefig(
    f"{folder}/critical_r01.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()


### STEP 5: Plot bifurcation value of R02

# Dependence on R01
sigmas = [0.4, 0.6, 0.8]
r01s = np.linspace(1, 10, 100001)

fig = plt.figure()
for sigma in sigmas:
    c_r02 = [critical_r02(r01, sigma) for r01 in r01s]
    plt.plot(r01s, c_r02, label=rf"$\sigma={sigma}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2}^{2\to 1}$", fontsize=fontsize_text)
plt.xlim(r01s[0], r01s[-1])
plt.ylim(0, 4)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/critical_r02_r01.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()

# Dependence on sigma
sigmas = np.linspace(0, 1, 100001)
r01s = [1.5, 3, 4.5]

fig = plt.figure()
for r01 in r01s:
    c_r02 = [critical_r02(r01, sigma) for sigma in sigmas]
    plt.plot(sigmas, c_r02, label=rf"$R_{{0,1}}={r01}$")

fig.legend(
    fontsize=fontsize_text,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=1,
)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,2}^{2\to 1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(sigmas[0], sigmas[-1])
plt.ylim(0, r01s[-1])
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/critical_r02_sigma.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()

# Contour plot
r01s = np.linspace(1, 4, 401)
sigmas = np.linspace(0, 1, 401)
critical_r02s = np.zeros((len(sigmas), len(r01s)))
for i, sigma in enumerate(sigmas):
    critical_r02s[i] = critical_r02(r01s, sigma)

levels = np.linspace(r01s[0], r01s[-1], 13)
cr = plt.contourf(sigmas, r01s, critical_r02s.T, levels=levels)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$R_{0,1}$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
cbar = plt.colorbar(cr)
cbar.set_label(r"$R_{0,2}^{2\to 1}$", size=fontsize_text)
cbar.ax.tick_params(labelsize=fontsize_ticks)
plt.savefig(
    f"{folder}/critical_r02.{fileformat}", dpi=dpi, bbox_inches="tight"
)
plt.show()
