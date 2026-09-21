"""
Computes the invasion reproduction number and plots the invasion regions.
"""

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
folder = "../plots"
alpha = 0.4
fontsize_text = 20
fontsize_ticks = 16

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
    # plt.title(rf"Stability regions for $\sigma$={round(sigma, 2)}")
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel(r"$R_{0,1}$", fontsize=fontsize_text)
    plt.ylabel(r"$R_{0,2}$", fontsize=fontsize_text)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{folder}/stability_region_sigma={round(sigma,2)}.jpg", dpi=dpi
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
    # plt.title(
    #     rf"Invasion reproduction number $R_{{0,1\to 2}}$ "
    #     rf"for $\sigma={round(sigma, 2)}$"
    # )
    plt.tight_layout()
    plt.savefig(
        f"{folder}/Invasion_reproduction_number_sigma={round(sigma,2)}.jpg",
        dpi=dpi,
    )
    plt.show()
    plt.close()


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
    f"{folder}/Stability_region_legend.jpg",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close(legend_fig)


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
    f"{folder}/Invasion_reproduction_number_colorbar.jpg",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close(colorbar_fig)


# # Compute and plot critical sigmas
# critical_sigmas = np.zeros((len(r01), len(r01)))
# for i, r01 in enumerate(r01):
#     critical_sigmas[i] = critical_sigma(r02, r01)

# cs = plt.contourf(r02, r02, critical_sigmas.T)
# plt.xlabel(r"$R_{0,1}$")
# plt.ylabel(r"$R_{0,2}$")
# # plt.title("Critical Sigma")
# cbar = plt.colorbar(cs)
# cbar.set_label(r"$\sigma^*_{1\to2}$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_sigma.jpg", dpi=dpi)
# plt.show()

# # Compute and plot critical R01
# sigmas = np.linspace(0, 1, 101)
# critical_r01s = np.zeros((len(sigmas), len(r02)))
# for i, sigma in enumerate(sigmas):
#     critical_r01s[i] = critical_r01(r02, sigma)

# cr = plt.contourf(sigmas, r02, critical_r01s.T)
# plt.xlabel(r"$\sigma$")
# plt.ylabel(r"$R_{0,2}$")
# # plt.title(r"Critical $R_0^1$")
# cbar = plt.colorbar(cr)
# cbar.set_label(r"$R_{0,1}^{1\to2}$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r01.jpg", dpi=dpi)
# plt.show()

# # Compute and plot critical R02
# sigmas = np.linspace(0, 1, 101)
# critical_r02s = np.zeros((len(sigmas), len(r02)))
# for i, sigma in enumerate(sigmas):
#     critical_r02s[i] = critical_r02(r02, sigma)

# levels = np.linspace(1.0, 20, 20)
# cr = plt.contourf(sigmas, r02, critical_r02s.T, levels=levels, extend="max")
# plt.xlabel(r"$\sigma$")
# plt.ylabel(r"$R_{0,1}$")
# # plt.title(r"Critical $R_0^1$")
# cbar = plt.colorbar(cr)
# cbar.set_label(r"$R_{0,2}^{1\to2}$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r02.jpg", dpi=dpi)
# plt.show()

# # Plot invasion reproduction number for combinations of parameters

# r01s = np.linspace(1, 4, 201)
# r02s = [4, 8]
# sigmas = [0.4, 0.8]

# for r02 in r02s:
#     for sigma in sigmas:
#         invasion_r0 = [
#             invasion_reproduction_number(r01, r02, sigma) for r01 in r01s
#         ]
#         plt.plot(
#             r01s, invasion_r0, label=rf"$R_{{0,2}}={r02},\,\sigma={sigma}$"
#         )

# plt.legend()
# plt.xlabel(r"$R_{0,1}$")
# plt.ylabel(r"$R_{0,1\to2}$")
# plt.axhline(1, color="grey", linestyle="dashed")
# plt.tight_layout()
# plt.savefig(f"{folder}/Invasion_reproduction_number_r01.jpg", dpi=dpi)
# plt.show()

# r01s = [1.5, 3]
# r02s = [4, 8]
# sigmas = np.linspace(0, 1, 201)

# for r01 in r01s:
#     for r02 in r02s:
#         invasion_r0 = [
#             invasion_reproduction_number(r01, r02, sigma) for sigma in sigmas
#         ]
#         plt.plot(
#             sigmas,
#             invasion_r0,
#             label=rf"$R_{{0,1}}={r01},\,R_{{0,2}}={r02}$",
#         )

# plt.legend()
# plt.xlabel(r"$\sigma$")
# plt.ylabel(r"$R_{0,1\to2}$")
# plt.axhline(1, color="grey", linestyle="dashed")
# plt.tight_layout()
# plt.savefig(f"{folder}/Invasion_reproduction_number_sigma.jpg", dpi=dpi)
# plt.show()

# r01s = [1.5, 3]
# r02s = np.linspace(1, 8, 201)
# sigmas = [0.4, 0.8]

# for r01 in r01s:
#     for sigma in sigmas:
#         invasion_r0 = [
#             invasion_reproduction_number(r01, r02, sigma) for r02 in r02s
#         ]
#         plt.plot(
#             r02s, invasion_r0, label=rf"$R_{{0,1}}={r01},\,\sigma={sigma}$"
#         )

# plt.legend()
# plt.xlabel(r"$R_{0,2}$")
# plt.ylabel(r"$R_{0,1\to2}$")
# plt.axhline(1, color="grey", linestyle="dashed")
# plt.tight_layout()
# plt.savefig(f"{folder}/Invasion_reproduction_number_r02.jpg", dpi=dpi)
# plt.show()

# r01s = [1.5, 3, 4.5]
# r02s = np.linspace(1, 10, 1001)
# for r01 in r01s:
#     c_sigma = [critical_sigma(r01, r02) for r02 in r02s]
#     plt.plot(r02s, c_sigma, label=rf"$R_{{0,1}}={r01}$")

# plt.legend()
# plt.xlabel(r"$R_{0,2}$")
# plt.ylabel(r"$\sigma_{1\to2}^*$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_sigma_r02.jpg", dpi=dpi)
# plt.show()

# r02s = [4, 6, 8]
# r01s = np.linspace(1, 4, 201)
# for r02 in r02s:
#     c_sigma = [critical_sigma(r01, r02) for r01 in r01s]
#     plt.plot(r01s, c_sigma, label=rf"$R_{{0,2}}={r02}$")

# plt.legend()
# plt.xlabel(r"$R_{0,1}$")
# plt.ylabel(r"$\sigma_{1\to2}^*$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_sigma_r01.jpg", dpi=dpi)
# plt.show()

# sigmas = [0.4, 0.6, 0.8]
# r02s = np.linspace(1, 10, 201)
# for sigma in sigmas:
#     c_r01 = [critical_r01(r02, sigma) for r02 in r02s]
#     plt.plot(r02s, c_r01, label=rf"$\sigma={sigma}$")

# plt.legend()
# plt.xlabel(r"$R_{0,2}$")
# plt.ylabel(r"$R_{0,1}^{1\to2}$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r01_r02.jpg", dpi=dpi)
# plt.show()

# sigmas = np.linspace(0, 1, 201)
# r02s = [4, 6, 8]
# for r02 in r02s:
#     c_r01 = [critical_r01(r02, sigma) for sigma in sigmas]
#     plt.plot(sigmas, c_r01, label=rf"$R_{{0,2}}={r02}$")

# plt.legend()
# plt.xlabel(r"$\sigma$")
# plt.ylabel(r"$R_{0,1}^{1\to2}$")
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r01_sigma.jpg", dpi=dpi)
# plt.show()

# sigmas = [0.4, 0.6, 0.8]
# r01s = np.linspace(1, 10, 100001)
# for sigma in sigmas:
#     c_r02 = [critical_r02(r01, sigma) for r01 in r01s]
#     plt.plot(r01s, c_r02, label=rf"$\sigma={sigma}$")
#     # plt.semilogy(r01s, c_r02, label=rf"$\sigma={sigma}$")

# plt.legend()
# plt.xlabel(r"$R_{0,1}$")
# plt.ylabel(r"$R_{0,2}^{1\to2}$")
# plt.ylim(top=20, bottom=0)
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r02_r01.jpg", dpi=dpi)
# plt.show()

# sigmas = np.linspace(0, 1, 100001)
# r01s = [1.5, 3, 4.5]
# for r01 in r01s:
#     c_r02 = [critical_r02(r01, sigma) for sigma in sigmas]
#     plt.plot(sigmas, c_r02, label=rf"$R_{{0,1}}={r01}$")
#     # plt.semilogy(sigmas, c_r02, label=rf"$R_{{0,1}}={r01}$")

# plt.legend()
# plt.xlabel(r"$\sigma$")
# plt.ylabel(r"$R_{0,2}^{1\to2}$")
# plt.ylim(top=20, bottom=0)
# plt.tight_layout()
# plt.savefig(f"{folder}/Critical_r02_sigma.jpg", dpi=dpi)
# plt.show()
