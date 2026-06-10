"""
Computes the invasion reproduction number and plots the invasion regions.
"""

import matplotlib.pyplot as plt
import numpy as np

from cross_immunity_model import (
    critical_r01,
    critical_r02,
    critical_sigma,
    invasion_reproduction_number,
)


# Set parameters
dpi = 500
folder = "../plots"

r02 = np.linspace(1, 4, 201)
sigmas = np.linspace(0, 1, 11)

alpha = 0.4

for sigma in sigmas:
    # Compute and plot invasion regions
    r01 = critical_r01(r02, sigma)
    if sigma < 1:
        plt.plot(r02, r01, "red")
        plt.plot(r01, r02, "red")
    plt.vlines(1, 0, 1, color="blue")
    plt.hlines(1, 0, 1, color="blue")
    plt.vlines(1, 1, r02[-1], color="purple")
    plt.hlines(1, 1, r02[-1], color="purple")
    plt.fill_between(
        r02,
        r02,
        r01,
        color="red",
        alpha=alpha,
        edgecolor="none",
        label="Both invasions possible",
    )
    plt.fill_betweenx(
        r02, r02, r01, color="red", alpha=alpha, edgecolor="none"
    )
    plt.fill_between(
        r02,
        r01,
        np.ones_like(r02),
        color="purple",
        alpha=alpha,
        edgecolor="none",
        label="One invasion possible",
    )
    plt.fill_betweenx(
        r02,
        r01,
        np.ones_like(r02),
        color="purple",
        alpha=alpha,
        edgecolor="none",
    )
    plt.fill_between(r02, 1, 0, color="blue", alpha=alpha)
    plt.fill_betweenx(r02, 1, 0, color="blue", alpha=alpha)
    plt.fill_between([0, 1], 0, 1, color="black", alpha=alpha)
    plt.xlim(0, r02[-1])
    plt.ylim(0, r02[-1])
    plt.title(rf"Invasion regions for $\sigma$={round(sigma, 1)}")
    plt.xlabel(r"$R_{0,1}$")
    plt.ylabel(r"$R_{0,2}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{folder}/Invasion_region_sigma={round(sigma,1)}.jpg", dpi=dpi
    )
    plt.show()

    # Compute and plot invasion reproduction number
    invasion_r0 = np.zeros((len(r02), len(r02)))
    for i, r01 in enumerate(r02):
        invasion_r0[i] = invasion_reproduction_number(r01, r02, sigma)

    ir = plt.contourf(r02, r02, invasion_r0.T)
    plt.xlabel(r"$R_{0,1}$")
    plt.ylabel(r"$R_{0,2}$")
    plt.title(
        rf"Invasion reproduction number $R_0^{{1\to 2}}$ "
        rf"for $\sigma={round(sigma, 2)}$"
    )
    cbar = plt.colorbar(ir)
    cbar.set_label(r"$R_0^{1\to 2}$")
    plt.tight_layout()
    plt.savefig(
        f"{folder}/Invasion_reproduction_number_sigma={round(sigma,1)}.jpg",
        dpi=dpi,
    )
    plt.show()

# Compute and plot critical sigmas
critical_sigmas = np.zeros((len(r02), len(r02)))
for i, r01 in enumerate(r02):
    critical_sigmas[i] = critical_sigma(r01, r02)

cs = plt.contourf(r02, r02, critical_sigmas.T)
plt.xlabel(r"$R_{0,1}$")
plt.ylabel(r"$R_{0,2}$")
# plt.title("Critical Sigma")
cbar = plt.colorbar(cs)
cbar.set_label(r"$\sigma^*_{1\to2}$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_sigma.jpg", dpi=dpi)
plt.show()

# Compute and plot critical R01
sigmas = np.linspace(0, 1, 101)
critical_r01s = np.zeros((len(sigmas), len(r02)))
for i, sigma in enumerate(sigmas):
    critical_r01s[i] = critical_r01(r02, sigma)

cr = plt.contourf(sigmas, r02, critical_r01s.T)
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$R_{0,2}$")
# plt.title(r"Critical $R_0^1$")
cbar = plt.colorbar(cr)
cbar.set_label(r"$R_{0,1}^{1\to2}$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r01.jpg", dpi=dpi)
plt.show()

# Compute and plot critical R02
sigmas = np.linspace(0, 1, 101)
critical_r02s = np.zeros((len(sigmas), len(r02)))
for i, sigma in enumerate(sigmas):
    critical_r02s[i] = critical_r02(r02, sigma)

levels = np.linspace(1.0, 20, 20)
cr = plt.contourf(sigmas, r02, critical_r02s.T, levels=levels, extend="max")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$R_{0,1}$")
# plt.title(r"Critical $R_0^1$")
cbar = plt.colorbar(cr)
cbar.set_label(r"$R_{0,2}^{1\to2}$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r02.jpg", dpi=dpi)
plt.show()

# Plot invasion reproduction number for combinations of parameters

r01s = np.linspace(1, 4, 201)
r02s = [4, 8]
sigmas = [0.4, 0.8]

for r02 in r02s:
    for sigma in sigmas:
        invasion_r0 = [
            invasion_reproduction_number(r01, r02, sigma) for r01 in r01s
        ]
        plt.plot(
            r01s, invasion_r0, label=rf"$R_{{0,2}}={r02},\,\sigma={sigma}$"
        )

plt.legend()
plt.xlabel(r"$R_{0,1}$")
plt.ylabel(r"$R_{0,1\to2}$")
plt.axhline(1, color="grey", linestyle="dashed")
plt.tight_layout()
plt.savefig(f"{folder}/Invasion_reproduction_number_r01.jpg", dpi=dpi)
plt.show()

r01s = [1.5, 3]
r02s = [4, 8]
sigmas = np.linspace(0, 1, 201)

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

plt.legend()
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$R_{0,1\to2}$")
plt.axhline(1, color="grey", linestyle="dashed")
plt.tight_layout()
plt.savefig(f"{folder}/Invasion_reproduction_number_sigma.jpg", dpi=dpi)
plt.show()

r01s = [1.5, 3]
r02s = np.linspace(1, 8, 201)
sigmas = [0.4, 0.8]

for r01 in r01s:
    for sigma in sigmas:
        invasion_r0 = [
            invasion_reproduction_number(r01, r02, sigma) for r02 in r02s
        ]
        plt.plot(
            r02s, invasion_r0, label=rf"$R_{{0,1}}={r01},\,\sigma={sigma}$"
        )

plt.legend()
plt.xlabel(r"$R_{0,2}$")
plt.ylabel(r"$R_{0,1\to2}$")
plt.axhline(1, color="grey", linestyle="dashed")
plt.tight_layout()
plt.savefig(f"{folder}/Invasion_reproduction_number_r02.jpg", dpi=dpi)
plt.show()

r01s = [1.5, 3, 4.5]
r02s = np.linspace(1, 10, 1001)
for r01 in r01s:
    c_sigma = [critical_sigma(r01, r02) for r02 in r02s]
    plt.plot(r02s, c_sigma, label=rf"$R_{{0,1}}={r01}$")

plt.legend()
plt.xlabel(r"$R_{0,2}$")
plt.ylabel(r"$\sigma_{1\to2}^*$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_sigma_r02.jpg", dpi=dpi)
plt.show()

r02s = [4, 6, 8]
r01s = np.linspace(1, 4, 201)
for r02 in r02s:
    c_sigma = [critical_sigma(r01, r02) for r01 in r01s]
    plt.plot(r01s, c_sigma, label=rf"$R_{{0,2}}={r02}$")

plt.legend()
plt.xlabel(r"$R_{0,1}$")
plt.ylabel(r"$\sigma_{1\to2}^*$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_sigma_r01.jpg", dpi=dpi)
plt.show()

sigmas = [0.4, 0.6, 0.8]
r02s = np.linspace(1, 10, 201)
for sigma in sigmas:
    c_r01 = [critical_r01(r02, sigma) for r02 in r02s]
    plt.plot(r02s, c_r01, label=rf"$\sigma={sigma}$")

plt.legend()
plt.xlabel(r"$R_{0,2}$")
plt.ylabel(r"$R_{0,1}^{1\to2}$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r01_r02.jpg", dpi=dpi)
plt.show()

sigmas = np.linspace(0, 1, 201)
r02s = [4, 6, 8]
for r02 in r02s:
    c_r01 = [critical_r01(r02, sigma) for sigma in sigmas]
    plt.plot(sigmas, c_r01, label=rf"$R_{{0,2}}={r02}$")

plt.legend()
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$R_{0,1}^{1\to2}$")
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r01_sigma.jpg", dpi=dpi)
plt.show()

sigmas = [0.4, 0.6, 0.8]
r01s = np.linspace(1, 10, 100001)
for sigma in sigmas:
    c_r02 = [critical_r02(r01, sigma) for r01 in r01s]
    plt.plot(r01s, c_r02, label=rf"$\sigma={sigma}$")
    # plt.semilogy(r01s, c_r02, label=rf"$\sigma={sigma}$")

plt.legend()
plt.xlabel(r"$R_{0,1}$")
plt.ylabel(r"$R_{0,2}^{1\to2}$")
plt.ylim(top=20, bottom=0)
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r02_r01.jpg", dpi=dpi)
plt.show()

sigmas = np.linspace(0, 1, 100001)
r01s = [1.5, 3, 4.5]
for r01 in r01s:
    c_r02 = [critical_r02(r01, sigma) for sigma in sigmas]
    plt.plot(sigmas, c_r02, label=rf"$R_{{0,1}}={r01}$")
    # plt.semilogy(sigmas, c_r02, label=rf"$R_{{0,1}}={r01}$")

plt.legend()
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$R_{0,2}^{1\to2}$")
plt.ylim(top=20, bottom=0)
plt.tight_layout()
plt.savefig(f"{folder}/Critical_r02_sigma.jpg", dpi=dpi)
plt.show()
