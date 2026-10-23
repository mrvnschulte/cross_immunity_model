"""
This script checks whether the algorithm to compute the coexisting variants
really finds a stable equilibrium.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cross_immunity_model import (
    eigenvalues_linearized_system,
    find_stable_multi_variant_equilibrium,
    sample_random_parameters,
)


def load_stability_data(
    csv_filename: str | Path,
) -> list[dict[str, str]]:
    """
    Load stability results from a CSV file.

    Parameters
    ----------
    csv_filename : str or pathlib.Path
        Path of the CSV file.

    Returns
    -------
    rows : list of dict
        CSV rows.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV file is empty or misses required columns.
    """
    csv_path = Path(csv_filename)

    try:
        csv_file = csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Set recompute = True to create it."
        ) from error

    with csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")

        required_columns = {
            "n_variants",
            "n_coexisting_variants",
            "max_real_part",
            "convergence",
        }

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV file {csv_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file contains no data: {csv_path}")

    return rows


def print_stability_summary(
    n_variants: int,
    csv_rows: list[dict[str, str]],
) -> None:
    """
    Print summary statistics for one number of variants.

    Only rows with convergence equal to one are used for the reported
    statistics.

    Parameters
    ----------
    n_variants : int
        Number of variants.
    csv_rows : list of dict
        Rows read from the corresponding CSV file.
    """
    converged_rows = [row for row in csv_rows if int(row["convergence"]) == 1]

    print()
    print(f"{n_variants} variants")
    print(f"  Stored parameter sets: {len(csv_rows)}")
    print(f"  Converged parameter sets: {len(converged_rows)}")

    if not converged_rows:
        print("  Mean number of coexisting variants: not available")
        print("  Largest maximal real part of eigenvalues: not available")
        return

    n_coexisting_variants = np.array(
        [int(row["n_coexisting_variants"]) for row in converged_rows],
        dtype=float,
    )

    max_real_parts = [
        float(row["max_real_part"])
        for row in converged_rows
        if row["max_real_part"] not in ("", "None", None)
    ]

    print(
        "  Mean number of coexisting variants: "
        f"{np.mean(n_coexisting_variants):.2f}"
    )

    print(
        "  Mean fraction of variants persisting: "
        f"{np.mean(n_coexisting_variants)/n_variants:.2f}"
    )

    if max_real_parts:
        print(
            "  Largest maximal real part of eigenvalues: "
            f"{max(max_real_parts):.3e}"
        )
    else:
        print("  Largest maximal real part of eigenvalues: not computed")


def check_stable_variants_algorithm(
    N_parameters: int,
    n_variants: int,
    beta_mean: float,
    beta_std: float,
    gamma_mean: float,
    gamma_std: float,
    sigma_alpha: float,
    sigma_beta: float,
    b: float,
    filename: str | None = None,
    csv_filename: str | None = None,
    compute_max_real_part: bool = True,
    dpi: int = 500,
    verbose_step: int | None = None,
) -> tuple[list[float | None], list[int]]:
    """
    Check whether the stable-equilibrium algorithm converges correctly.

    Generates ``N_parameters`` independent parameter sets for a model with
    ``n_variants`` variants and, for each set:

    1. Samples ``(beta, gamma, sigma)`` via ``sample_random_parameters``.
    2. Locates the stable equilibrium with
       ``find_stable_multi_variant_equilibrium``.
    3. Computes the eigenvalues of the Jacobian at that equilibrium via
       ``eigenvalues_linearized_system``.
    4. Records the maximal real part of those eigenvalues.

    A scatter plot of the maximal real parts against the parameter-set index
    is produced. All maximal real parts should be strictly negative if the
    algorithm converges to a stable equilibrium in every case.

    Parameters
    ----------
    N_parameters : int
        Number of parameter sets to be generated.
    n_variants : int
        Number of variants in the model.
    beta_mean : float
        Mean of the transmission rate.
    beta_std : float
        Standard deviation of the transmission rate.
    gamma_mean : float
        Mean of the recovery rate.
    gamma_std : float
        Standard deviation of the recovery rate.
    sigma_alpha : float
        Alpha parameter of the Beta distribution for cross-immunity factors.
    sigma_beta : float
        Beta parameter of the Beta distribution for cross-immunity factors.
    b : float
        Natural birth and death rate.
    filename : str or None, optional
        Path at which to save the figure. If None the figure is not saved.
        The default is None.
    csv_filename : str or None, optional
        Path at which to save the data. If None the data is not saved.
        The default is None.
    compute_max_real_part : bool, optional
        If True, computes the maximal real part of the Jacobian eigenvalues.
        This can be expensive for many variants. The default is True.
    dpi : int, optional
        Resolution in dots per inch used when saving the figure.
        The default is 500.
    verbose_step : int or None, optional
        If not None, a progress message is printed every ``verbose_step``
        iterations. The default is None.

    Returns
    -------
    max_real_parts : list[float | None]
        Maximal real part for each parameter set. Entries are None if
        ``compute_max_real_part=False``.
    n_coexisting_variants : list[int]
        Number of variants in the equilibrium for each parameter set.

    Side Effects
    ------------
    Displays a scatter plot via ``plt.show()``.
    Writes a JPEG file to ``filename`` when ``filename`` is not None.
    Prints a warning to stdout for each parameter set where
    ``find_stable_multi_variant_equilibrium`` does not converge
    (``convergence != 1``).

    Notes
    -----
    A maximal real part strictly below zero confirms asymptotic stability of
    the found equilibrium. Values at or above zero indicate either that the
    algorithm converged to an unstable equilibrium or that numerical errors
    are present.

    The ``i``-th parameter set is seeded with ``seed=i``, so results are
    fully reproducible.
    """
    max_real_parts = []
    n_coexisting_variants = []
    csv_rows = []
    for i in range(N_parameters):
        if verbose_step is not None:
            if (i + 1) % verbose_step == 0:
                print(f"Compute set {i+1}/{N_parameters}")
        beta, gamma, sigma = sample_random_parameters(
            n_variants,
            beta_mean,
            beta_std,
            gamma_mean,
            gamma_std,
            sigma_alpha,
            sigma_beta,
            seed=i,
        )
        coexisting_variants, I_variant_eq, x_eq, convergence = (
            find_stable_multi_variant_equilibrium(
                beta, gamma, sigma, b, atol=1e-4, use_inv_formulation=True
            )
        )
        if convergence != 1:
            print(
                f"Note: For seed {i} a stable equilibrium has not been found, probably due to not finding the correct equilibrium"
            )
            if not compute_max_real_part:
                continue
        if compute_max_real_part:
            _, max_real_part, _ = eigenvalues_linearized_system(
                x_eq,
                beta,
                gamma,
                sigma,
                b,
            )
            max_real_part = float(max_real_part)
        else:
            max_real_part = None
        n_coexisting = len(coexisting_variants)
        max_real_parts.append(max_real_part)
        n_coexisting_variants.append(n_coexisting)

        csv_rows.append(
            {
                "parameter_set": i,
                "seed": i,
                "n_variants": n_variants,
                "n_coexisting_variants": n_coexisting,
                "coexisting_variants": ";".join(
                    str(variant + 1) for variant in coexisting_variants
                ),
                "max_real_part": max_real_part,
                "convergence": convergence,
            }
        )
    if csv_filename is not None:
        csv_path = Path(csv_filename)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "parameter_set",
                    "seed",
                    "n_variants",
                    "n_coexisting_variants",
                    "coexisting_variants",
                    "max_real_part",
                    "convergence",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"csv file saved: {csv_path}")

    if compute_max_real_part:
        plt.figure()
        plt.scatter(range(N_parameters), max_real_parts)
        plt.axhline(color="grey", linestyle="dotted")
        plt.xlabel("Index of parameter set", fontsize=fontsize_text)
        plt.ylabel("Maximal real part of\neigenvalues", fontsize=fontsize_text)
        plt.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )
        plt.gca().yaxis.get_offset_text().set_fontsize(fontsize_ticks)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        if filename is not None:
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()

    return max_real_parts, n_coexisting_variants


dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
plot_folder = "../plots"
data_folder = "../data"
fileformat = "pdf"
verbose = True
verbose_step = 10

# If True, recompute results, save CSV files, and show the existing plots.
# If False, load existing CSV files and print summaries only.
recompute = False

# Parameter values
beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0
sigma_alpha = 9
sigma_beta = 1
b = 0.01
N_parameters = 1000
check_variants = [2, 3, 4, 5, 6, 7]  # , 7, 8, 9, 10]
compute_max_real_part = True

# Main part
for n_variants in check_variants:
    filename = (
        f"{plot_folder}/numerical_stability_{n_variants}_variants.{fileformat}"
    )
    csv_filename = (
        f"{data_folder}/numerical_stability_{n_variants}_variants.csv"
    )
    if n_variants > 6:
        compute_max_real_part = False
    if recompute:
        check_stable_variants_algorithm(
            N_parameters,
            n_variants,
            beta_mean,
            beta_std,
            gamma_mean,
            gamma_std,
            sigma_alpha,
            sigma_beta,
            b,
            filename=filename,
            csv_filename=csv_filename,
            compute_max_real_part=compute_max_real_part,
            dpi=dpi,
            verbose_step=verbose_step,
        )

        if verbose:
            print(f"Finished computation for {n_variants} variants.")

    csv_rows = load_stability_data(csv_filename)

    if verbose and not recompute:
        print(f"CSV file loaded: {csv_filename}")

    print_stability_summary(
        n_variants,
        csv_rows,
    )
