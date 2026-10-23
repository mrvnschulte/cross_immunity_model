"""
Solves some example multi-variant systems and plots them.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_with_matrices,
    invasion_reproduction_number_multiple_variants,
    infectious_per_variant,
)

# Solve a two-variant model
# t0 = 0
# tf = 1000
# x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])
# beta = [0.3, 0.35]
# gamma = [0.1, 0.1]
# sigma = np.array([[1, 0.8], [0.8, 1]])
# b = 0.01

# res12 = solve_ivp(
#     multi_variant_model,
#     (t0, tf),
#     x0,
#     t_eval=np.arange(tf),
#     args=(beta, gamma, sigma, b),
# )

# plt.plot(
#     res12.t,
#     res12.y.T,
#     label=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12"],
# )
# inv_r0 = [
#     invasion_reproduction_number_multiple_variants(x, beta, gamma, b)
#     for x in res12.y.T
# ]
# plt.plot(res12.t, inv_r0, label=["inv_r01", "inv_r02"])
# plt.xlabel("Time")
# plt.ylabel("Compartment size")
# plt.legend()
# plt.show()

# Solve a 1-variant SIR model using multi-variant model
vaccinated = 25
x0 = np.array([(78 - vaccinated) / 80, 0 / 80, 2 / 80])
beta = [0.15]
gamma = [0.1]
sigma = np.array([[1]])
t0 = 0
tf = 200
population = 80
fontsize_text = 16
fontsize_ticks = 14
t_plot = [0, 200]

res1 = solve_ivp(
    multi_variant_model_for_loops,
    (t0, tf),
    x0,
    t_eval=np.arange(tf + 1),
    args=(beta, gamma, sigma),
)

death_rate = 0.5 * (25 - min(25, vaccinated)) / (80 - vaccinated) + 0.05 * (
    55 - max(vaccinated - 25, 0)
) / (80 - vaccinated)

for time in t_plot:
    print(f"S({time})={res1.y[0][time] * population}")
    print(f"I({time})={res1.y[2][time] * population}")
    print(f"R({time})={res1.y[1][time] * population}")
    # plt.plot(
    #     res1.t[:time],
    #     res1.y[0][:time] * population,
    #     label="S",
    #     color="#B9C2C9",
    # )
    plt.plot(
        res1.t[:time],
        res1.y[2][:time] * population,
        label="I",
        color="#D94A4A",
    )
    plt.plot(
        res1.t[:time],
        res1.y[1][:time] * population * death_rate,
        label="D",
        color="#000000",
    )
    plt.xlabel("Time", fontsize=fontsize_text)
    plt.ylabel("Compartment size", fontsize=fontsize_text)
    plt.legend(fontsize=fontsize_text)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(t0, tf)
    plt.ylim(bottom=0, top=10)
    plt.show()


# =============================================================================
# Effect of vaccination on the number of deaths
# =============================================================================

t0 = 0
tf = 200
population = 80

beta = [0.15]
gamma = [0.1]
sigma = np.array([[1.0]])

# Two persons are initially infectious.
initial_infectious = 2

# At most 78 persons can be vaccinated, since two are initially infectious.
vaccinated_persons = np.arange(0, population - initial_infectious + 1)

deaths_at_tf = np.zeros(len(vaccinated_persons))
final_infections = np.zeros(len(vaccinated_persons))
death_rates = np.zeros(len(vaccinated_persons))

for index, vaccinated in enumerate(vaccinated_persons):
    # Vaccination strategy:
    # First vaccinate the 25 persons with high mortality risk.
    # Then vaccinate the remaining 55 persons with low mortality risk.
    #
    # The death rate is the average infection fatality rate among the
    # unvaccinated population.
    death_rate = (
        0.5 * (25 - min(25, vaccinated))
        + 0.05 * (55 - max(vaccinated - 25, 0))
    ) / (population - vaccinated)

    death_rates[index] = death_rate

    # State ordering: [S, R, I].
    # Vaccinated persons are excluded from the model population.
    x0 = np.array(
        [
            (population - vaccinated - initial_infectious) / population,
            0.0,
            initial_infectious / population,
        ]
    )

    res = solve_ivp(
        multi_variant_model_for_loops,
        (t0, tf),
        x0,
        t_eval=np.arange(t0, tf + 1),
        args=(beta, gamma, sigma),
    )

    if not res.success:
        raise RuntimeError(
            f"Simulation failed for vaccinated = {vaccinated}: {res.message}"
        )

    # R(tf) is the fraction of the total original population that has
    # experienced and recovered from infection by time tf.
    recovered_at_tf = res.y[1, -1]

    # Final number of infected/recovered persons.
    final_infections[index] = recovered_at_tf * population

    # Expected number of deaths among infected persons.
    deaths_at_tf[index] = recovered_at_tf * population * death_rate

# =============================================================================
# Plot
# =============================================================================

fontsize_text = 16
fontsize_ticks = 14

fig = plt.figure()

plt.plot(
    vaccinated_persons,
    deaths_at_tf,
    color="#000000",
    linewidth=2.5,
)
plt.axvline(
    x=25,
    color="#7F7F7F",
    linestyle="--",
    linewidth=2,
    label="Risk-group persons",
)


plt.xlabel("Vaccinated persons", fontsize=fontsize_text)
plt.ylabel("Number of deaths", fontsize=fontsize_text)

plt.xlim(0, population - initial_infectious)
plt.ylim(bottom=0, top=10)
plt.legend(fontsize=fontsize_ticks)

plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)

plt.tight_layout()
plt.show()

# Solve a two-variant model
# x0 = np.array([78 / 80, 0 / 80, 1 / 80, 0, 1 / 80, 0, 0, 0, 0])
# beta = [0.15, 0.18]
# gamma = [0.1, 0.1]
# sigma = np.array([[1, 0.8], [0.8, 1]])
# t0 = 0
# tf = 200
# population = 80
# fontsize_text = 16
# fontsize_ticks = 14
# t_plot = [0, 50, 200]

# res1 = solve_ivp(
#     multi_variant_model_for_loops,
#     (t0, tf),
#     x0,
#     t_eval=np.arange(tf + 1),
#     args=(beta, gamma, sigma),
# )
# i_var = np.array(
#     [infectious_per_variant(res1.y.T[i]) for i in range(len(res1.y[0]))]
# )

# for time in t_plot:
#     print(f"I_1({time})={i_var[time] * population}")
#     print(f"I_2({time})={i_var[time] * population}")
#     print(f"x({time})={res1.y.T[time] * population}")
#     # plt.plot(
#     #     res1.t[:time],
#     #     infectious_per_variant(res1.y[0][:time] * population,
#     #     label="S",
#     #     color="#B9C2C9",
#     # )
#     plt.plot(
#         res1.t[:time],
#         i_var[:, 0][:time] * population,
#         label=r"$I_1$",
#         color="#D94A4A",
#     )
#     plt.plot(
#         res1.t[:time],
#         i_var[:, 1][:time] * population,
#         label=r"$I_2$",
#         color="#F28E2B",
#     )
#     plt.xlabel("Time", fontsize=fontsize_text)
#     plt.ylabel("Compartment size", fontsize=fontsize_text)
#     plt.legend(fontsize=fontsize_text)
#     plt.xticks(fontsize=fontsize_ticks)
#     plt.yticks(fontsize=fontsize_ticks)
#     plt.xlim(t0, tf)
#     plt.ylim(0, 8)
#     plt.show()


# # Solve three-variant model
# # t0 = 0
# # tf = 5000
# # beta_3 = 0.14
# # sigma_23 = 0.3
# # beta = [0.3, 0.35, beta_3]
# # gamma = [0.1, 0.1, 0.1]
# # b = 0.01
# # sigma_12 = 0.5
# # sigma = np.array(
# #     [
# #         [1, sigma_12, sigma_12 * sigma_23],
# #         [sigma_12, 1, sigma_23],
# #         [sigma_12 * sigma_23, sigma_23, 1],
# #     ]
# # )
# # n_variants = len(beta)
# # size = 2**n_variants
# # x0 = 0.001 * np.ones(3**n_variants)
# # x0[0] = 1 - np.sum(x0[1:])

# # res12 = solve_ivp(
# #     multi_variant_model,
# #     (t0, tf),
# #     x0,
# #     t_eval=np.arange(tf),
# #     args=(beta, gamma, sigma, b),
# # )

# # plt.plot(
# #     res12.t,
# #     res12.y.T,
# # )
# # plt.show()

# # # Solve two-variant model with matrices
# # beta = beta[1:]
# # gamma = gamma[1:]
# # sigma = np.array([[1, sigma_23], [sigma_23, 1]])
# # n_variants = len(beta)
# # size = 2**n_variants
# # V, G, T, S = construct_matrices(gamma, sigma)
# # x0 = 0.001 * np.ones(3**n_variants)
# # x0[0] = 1 - np.sum(x0[1:])

# # res12 = solve_ivp(
# #     multi_variant_model_with_matrices,
# #     (t0, tf),
# #     x0,
# #     t_eval=np.arange(tf),
# #     args=(beta, gamma, sigma, V, G, T, S, b),
# # )

# # plt.plot(
# #     res12.t,
# #     res12.y.T,
# # )
# # plt.show()

# # =============================================================================
# # One-variant model until t = 50, followed by a two-variant model
# # =============================================================================

# t0 = 0
# t_introduction = 110
# tf = 200

# population = 80
# beta_one = [0.15]
# gamma_one = [0.1]
# sigma_one = np.array([[1.0]])
# sigma_12 = 0.5

# # Initial state of the one-variant model:
# # State ordering for one variant: [S, R_1, I_1]
# x0_one = np.array(
#     [
#         78 / population,  # S
#         1 / population,  # R_1
#         1 / population,  # I_1
#     ]
# )

# # -----------------------------------------------------------------------------
# # Simulate the one-variant model until the introduction time
# # -----------------------------------------------------------------------------

# res_one = solve_ivp(
#     multi_variant_model_for_loops,
#     (t0, t_introduction),
#     x0_one,
#     t_eval=np.arange(t0, t_introduction + 1),
#     args=(beta_one, gamma_one, sigma_one),
# )

# # State at t = 50:
# # x_one_at_introduction = [S, R_1, I_1]
# x_one_at_introduction = res_one.y[:, -1]

# # -----------------------------------------------------------------------------
# # Construct the initial state of the two-variant model at t = 50
# # -----------------------------------------------------------------------------
# # State ordering for two variants:
# # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]

# seed_variant_2 = 1 / population

# x0_two = np.zeros(9)

# # Transfer the state of the one-variant model.
# x0_two[0] = x_one_at_introduction[0] - seed_variant_2  # S
# x0_two[1] = (1 - sigma_12) * x_one_at_introduction[1]  # R_1
# x0_two[2] = (1 - sigma_12) * x_one_at_introduction[2]  # I_1
# x0_two[5] = sigma_12 * x_one_at_introduction[1]
# x0_two[6] = sigma_12 * x_one_at_introduction[2]

# # Introduce the second variant.
# x0_two[4] = seed_variant_2  # I_2

# # Safety check: the population fractions should sum to one.
# assert np.isclose(np.sum(x0_two), 1.0), "Population is not conserved."

# # Parameters after the introduction of variant 2.
# beta_two = [0.15, 0.18]
# gamma_two = [0.1, 0.1]
# sigma_two = np.array(
#     [
#         [1.0, sigma_12],
#         [sigma_12, 1.0],
#     ]
# )

# # -----------------------------------------------------------------------------
# # Simulate the two-variant model from t = 50 onwards
# # -----------------------------------------------------------------------------

# res_two = solve_ivp(
#     multi_variant_model_for_loops,
#     (t_introduction, tf),
#     x0_two,
#     t_eval=np.arange(t_introduction, tf + 1),
#     args=(beta_two, gamma_two, sigma_two),
# )

# # Infectious population fractions for both variants after t = 50.
# infectious_two = np.array(
#     [infectious_per_variant(res_two.y.T[i]) for i in range(res_two.y.shape[1])]
# )

# # I_1 before t = 50 equals the infected compartment of the one-variant model.
# infectious_one_before = res_one.y[2, :]

# # Combine I_1 trajectories while avoiding a duplicated value at t = 50.
# times_combined = np.concatenate(
#     [
#         res_one.t,
#         res_two.t[1:],
#     ]
# )

# infectious_variant_1 = np.concatenate(
#     [
#         infectious_one_before,
#         infectious_two[1:, 0],
#     ]
# )

# # I_2 is only defined after its introduction at t = 50.
# times_variant_2 = res_two.t
# infectious_variant_2 = infectious_two[:, 1]

# # -----------------------------------------------------------------------------
# # Plot
# # -----------------------------------------------------------------------------

# fontsize_text = 16
# fontsize_ticks = 14

# fig = plt.figure()

# plt.plot(
#     times_combined[:t_introduction],
#     infectious_variant_1[:t_introduction] * population,
#     label=r"$I_1$",
#     color="#D94A4A",
#     linewidth=2,
# )
# plt.axvline(
#     t_introduction,
#     color="black",
#     linestyle="--",
#     linewidth=1,
# )


# plt.xlabel("Time", fontsize=fontsize_text)
# plt.ylabel("Compartment size", fontsize=fontsize_text)

# plt.xlim(t0, tf)
# plt.ylim(bottom=0, top=8)

# plt.legend(fontsize=fontsize_text)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.tight_layout()
# plt.show()

# fig = plt.figure()

# plt.plot(
#     times_combined,
#     infectious_variant_1 * population,
#     label=r"$I_1$",
#     color="#D94A4A",
#     linewidth=2,
# )

# plt.plot(
#     times_variant_2,
#     infectious_variant_2 * population,
#     label=r"$I_2$",
#     color="#F28E2B",
#     linewidth=2,
# )

# # Optional: mark the introduction time.
# plt.axvline(
#     t_introduction,
#     color="black",
#     linestyle="--",
#     linewidth=1,
# )

# plt.xlabel("Time", fontsize=fontsize_text)
# plt.ylabel("Compartment size", fontsize=fontsize_text)

# plt.xlim(t0, tf)
# plt.ylim(bottom=0, top=8)

# plt.legend(fontsize=fontsize_text)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.tight_layout()
# plt.show()


# """
# Expected epidemic dynamics for a two-variant model with a stochastic
# mutation time, including 95% uncertainty intervals.

# The mutation intensity is assumed to be proportional to the number
# of individuals infectious with variant 1:

#     lambda(t) = mutation_rate * I_1(t)
# """

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.integrate import solve_ivp

# from cross_immunity_model import (
#     infectious_per_variant,
#     multi_variant_model_for_loops,
# )


# def weighted_quantile(values, quantiles, weights):
#     """Compute weighted empirical quantiles.

#     Parameters
#     ----------
#     values : np.ndarray
#         One-dimensional array of values.
#     quantiles : float or array-like
#         Quantiles in [0, 1].
#     weights : np.ndarray
#         Non-negative probability weights associated with values.

#     Returns
#     -------
#     np.ndarray
#         Weighted empirical quantiles.
#     """
#     values = np.asarray(values)
#     quantiles = np.atleast_1d(quantiles)
#     weights = np.asarray(weights)

#     if len(values) != len(weights):
#         raise ValueError("values and weights must have equal length.")

#     if np.any(weights < 0):
#         raise ValueError("weights must be non-negative.")

#     if np.sum(weights) <= 0:
#         raise ValueError("The sum of weights must be positive.")

#     order = np.argsort(values)
#     values_sorted = values[order]
#     weights_sorted = weights[order]

#     cumulative_weights = np.cumsum(weights_sorted)
#     cumulative_weights /= cumulative_weights[-1]

#     indices = np.searchsorted(cumulative_weights, quantiles, side="left")
#     indices = np.clip(indices, 0, len(values_sorted) - 1)

#     return values_sorted[indices]


# # =============================================================================
# # Parameters
# # =============================================================================

# t0 = 0
# tf = 200
# dt = 1.0

# population = 80

# # Initial state of the one-variant model:
# # State ordering for one variant: [S, R_1, I_1]
# x0_one = np.array(
#     [
#         78 / population,  # S
#         1 / population,  # R_1
#         1 / population,  # I_1
#     ]
# )

# # Parameters before a mutation occurs.
# beta_one = [0.15]
# gamma_one = [0.1]
# sigma_one = np.array([[1.0]])

# # Parameters after the emergence of variant 2.
# beta_two = [0.15, 0.18]
# gamma_two = [0.1, 0.1]

# sigma_12 = 0.5
# sigma_two = np.array(
#     [
#         [1.0, sigma_12],
#         [sigma_12, 1.0],
#     ]
# )

# # Mutation parameter in lambda(t) = mutation_rate * I_1(t).
# mutation_rate = 0.5

# # The mutation seeds one individual infected with variant 2.
# seed_variant_2 = 1 / population

# # Time grid.
# times = np.arange(t0, tf + dt, dt)


# # =============================================================================
# # Simulate the one-variant model without mutation
# # =============================================================================

# res_one = solve_ivp(
#     multi_variant_model_for_loops,
#     (t0, tf),
#     x0_one,
#     t_eval=times,
#     args=(beta_one, gamma_one, sigma_one),
# )

# if not res_one.success:
#     raise RuntimeError(f"One-variant simulation failed: {res_one.message}")

# # In the one-variant model, index 2 corresponds to I_1.
# i1_no_mutation = res_one.y[2, :]


# # =============================================================================
# # Discrete distribution of the mutation time
# # =============================================================================
# #
# # For a time interval [t_k, t_{k+1}), use the hazard
# #
# #     lambda_k = mutation_rate * I_1(t_k).
# #
# # If S_k is the probability of no mutation until t_k, then
# #
# #     P(T_mut in [t_k, t_{k+1})) =
# #         S_k * (1 - exp(-lambda_k * dt)).
# #
# # The remaining probability mass is assigned to the event that no mutation
# # occurs before tf.
# # =============================================================================

# mutation_times = times[:-1]
# n_mutation_times = len(mutation_times)

# mutation_time_probabilities = np.zeros(n_mutation_times)
# survival_probability = 1.0

# for k in range(n_mutation_times):
#     hazard = mutation_rate * i1_no_mutation[k]

#     mutation_time_probabilities[k] = survival_probability * (
#         1 - np.exp(-hazard * dt)
#     )

#     survival_probability *= np.exp(-hazard * dt)

# probability_no_mutation = survival_probability
# probability_mutation = np.sum(mutation_time_probabilities)

# print(f"Probability of mutation until t = {tf}: {probability_mutation:.4f}")
# print(
#     f"Probability of no mutation until t = {tf}: "
#     f"{probability_no_mutation:.4f}"
# )
# print(
#     f"Total probability mass: "
#     f"{probability_mutation + probability_no_mutation:.8f}"
# )


# # =============================================================================
# # Simulate a two-variant trajectory for every possible mutation time
# # =============================================================================
# #
# # Each row of i1_trajectories and i2_trajectories represents one possible
# # realization:
# #
# # - row 0: no mutation occurs before tf;
# # - row k + 1: mutation occurs at mutation_times[k].
# # =============================================================================

# i1_trajectories = []
# i2_trajectories = []
# trajectory_weights = []

# # Scenario 0: no mutation.
# i1_trajectories.append(i1_no_mutation.copy())
# i2_trajectories.append(np.zeros_like(times, dtype=float))
# trajectory_weights.append(probability_no_mutation)

# for mutation_index, mutation_time in enumerate(mutation_times):
#     probability = mutation_time_probabilities[mutation_index]

#     # Ignore events with essentially zero probability.
#     if probability < 1e-14:
#         continue

#     # State of the one-variant system at the mutation time:
#     # [S, R_1, I_1]
#     x_one_at_mutation = res_one.y[:, mutation_index]

#     if x_one_at_mutation[0] < seed_variant_2:
#         raise ValueError(
#             "The susceptible population at the mutation time is smaller "
#             "than seed_variant_2."
#         )

#     # State ordering for two variants:
#     #
#     # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]
#     #
#     # Individuals protected against variant 2 due to infection with
#     # variant 1 are transferred to R_12 or I_1,12.
#     x0_two = np.zeros(9)

#     x0_two[0] = x_one_at_mutation[0] - seed_variant_2
#     x0_two[1] = (1 - sigma_12) * x_one_at_mutation[1]
#     x0_two[2] = (1 - sigma_12) * x_one_at_mutation[2]
#     x0_two[4] = seed_variant_2
#     x0_two[5] = sigma_12 * x_one_at_mutation[1]
#     x0_two[6] = sigma_12 * x_one_at_mutation[2]

#     assert np.isclose(
#         np.sum(x0_two), 1.0
#     ), "Population is not conserved during the state transformation."

#     # Simulate the two-variant system from mutation_time until tf.
#     times_after_mutation = times[mutation_index:]

#     res_two = solve_ivp(
#         multi_variant_model_for_loops,
#         (mutation_time, tf),
#         x0_two,
#         t_eval=times_after_mutation,
#         args=(beta_two, gamma_two, sigma_two),
#     )

#     if not res_two.success:
#         raise RuntimeError(
#             f"Two-variant simulation failed at t = {mutation_time}: "
#             f"{res_two.message}"
#         )

#     infectious_two = np.array(
#         [infectious_per_variant(state) for state in res_two.y.T]
#     )

#     # Construct the full trajectory on [t0, tf].
#     # Before the mutation, only variant 1 is present.
#     i1_path = i1_no_mutation.copy()
#     i2_path = np.zeros_like(times, dtype=float)

#     i1_path[mutation_index:] = infectious_two[:, 0]
#     i2_path[mutation_index:] = infectious_two[:, 1]

#     i1_trajectories.append(i1_path)
#     i2_trajectories.append(i2_path)
#     trajectory_weights.append(probability)

# # Convert lists to arrays.
# i1_trajectories = np.asarray(i1_trajectories)
# i2_trajectories = np.asarray(i2_trajectories)
# trajectory_weights = np.asarray(trajectory_weights)

# # Protect against minor numerical rounding errors.
# trajectory_weights /= np.sum(trajectory_weights)


# # =============================================================================
# # Expected trajectories
# # =============================================================================

# expected_i1 = np.average(
#     i1_trajectories,
#     axis=0,
#     weights=trajectory_weights,
# )

# expected_i2 = np.average(
#     i2_trajectories,
#     axis=0,
#     weights=trajectory_weights,
# )


# # =============================================================================
# # Pointwise 68% uncertainty intervals
# # =============================================================================

# lower_i1 = np.zeros_like(times, dtype=float)
# upper_i1 = np.zeros_like(times, dtype=float)

# lower_i2 = np.zeros_like(times, dtype=float)
# upper_i2 = np.zeros_like(times, dtype=float)

# for time_index in range(len(times)):
#     lower_i1[time_index], upper_i1[time_index] = weighted_quantile(
#         i1_trajectories[:, time_index],
#         quantiles=[0.16, 0.84],
#         weights=trajectory_weights,
#     )

#     lower_i2[time_index], upper_i2[time_index] = weighted_quantile(
#         i2_trajectories[:, time_index],
#         quantiles=[0.16, 0.84],
#         weights=trajectory_weights,
#     )


# # =============================================================================
# # Plot: expected dynamics and pointwise 95% uncertainty intervals
# # =============================================================================

# fontsize_text = 16
# fontsize_ticks = 14

# fig = plt.figure()

# # 68% uncertainty interval for I_1.
# plt.fill_between(
#     times,
#     lower_i1 * population,
#     upper_i1 * population,
#     color="#D94A4A",
#     alpha=0.20,
#     linewidth=0,
#     label=r"68 % interval for $I_1$",
# )

# # 68% uncertainty interval for I_2.
# plt.fill_between(
#     times,
#     lower_i2 * population,
#     upper_i2 * population,
#     color="#F28E2B",
#     alpha=0.20,
#     linewidth=0,
#     label=r"68 % interval for $I_2$",
# )


# # Expected numbers of infectious individuals.
# plt.plot(
#     times,
#     expected_i1 * population,
#     color="#D94A4A",
#     linewidth=2.5,
#     label=r"Expected $I_1$",
# )

# plt.plot(
#     times,
#     expected_i2 * population,
#     color="#F28E2B",
#     linewidth=2.5,
#     label=r"Expected $I_2$",
# )

# plt.xlabel("Time", fontsize=fontsize_text)
# plt.ylabel(
#     "Compartment size",
#     fontsize=fontsize_text,
# )

# plt.xlim(t0, tf)
# plt.ylim(bottom=0, top=8)

# plt.legend(
#     fontsize=fontsize_ticks,
#     ncol=2,
#     loc="upper right",
# )

# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.tight_layout()
# plt.show()


# """
# Expected epidemic dynamics for a two-variant model with a stochastic
# mutation time, including expected cumulative deaths and pointwise
# 68% uncertainty intervals.

# The mutation intensity is assumed to be proportional to the number of
# individuals infectious with variant 1:

#     lambda(t) = mutation_rate * I_1(t)

# Deaths are approximated as a fixed fraction of cumulative infections:

#     D(t) = death_rate * R_total(t) * population
# """

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.integrate import solve_ivp

# from cross_immunity_model import (
#     infectious_per_variant,
#     multi_variant_model_for_loops,
# )


# def weighted_quantile(values, quantiles, weights):
#     """Compute weighted empirical quantiles.

#     Parameters
#     ----------
#     values : np.ndarray
#         One-dimensional array of values.
#     quantiles : float or array-like
#         Quantiles in [0, 1].
#     weights : np.ndarray
#         Non-negative probability weights associated with values.

#     Returns
#     -------
#     np.ndarray
#         Weighted empirical quantiles.
#     """
#     values = np.asarray(values)
#     quantiles = np.atleast_1d(quantiles)
#     weights = np.asarray(weights)

#     if len(values) != len(weights):
#         raise ValueError("values and weights must have equal length.")

#     if np.any(weights < 0):
#         raise ValueError("weights must be non-negative.")

#     if np.sum(weights) <= 0:
#         raise ValueError("The sum of weights must be positive.")

#     order = np.argsort(values)
#     values_sorted = values[order]
#     weights_sorted = weights[order]

#     cumulative_weights = np.cumsum(weights_sorted)
#     cumulative_weights /= cumulative_weights[-1]

#     indices = np.searchsorted(cumulative_weights, quantiles, side="left")
#     indices = np.clip(indices, 0, len(values_sorted) - 1)

#     return values_sorted[indices]


# # =============================================================================
# # Parameters
# # =============================================================================

# t0 = 0
# tf = 200
# dt = 1.0

# population = 80

# # State ordering for the one-variant model:
# # [S, R_1, I_1]
# x0_one = np.array(
#     [
#         78 / population,  # Susceptible persons
#         1 / population,  # Recovered/protected persons
#         1 / population,  # Infectious persons
#     ]
# )

# # Parameters before mutation.
# beta_one = [0.15]
# gamma_one = [0.1]
# sigma_one = np.array([[1.0]])

# # Parameters after mutation.
# beta_two = [0.15, 0.18]
# gamma_two = [0.1, 0.1]

# sigma_12 = 0.5
# sigma_two = np.array(
#     [
#         [1.0, sigma_12],
#         [sigma_12, 1.0],
#     ]
# )

# # Mutation intensity:
# # lambda(t) = mutation_rate * I_1(t)
# mutation_rate = 0.5

# # The emergence of variant 2 seeds one infected individual.
# seed_variant_2 = 1 / population

# # Constant probability of death per infection.
# # Example: population consists of 25 high-risk persons with mortality 0.5
# # and 55 other persons with mortality 0.05.
# death_rate = (25 * 0.5 + 55 * 0.05) / population

# # Time grid.
# times = np.arange(t0, tf + dt, dt)


# # =============================================================================
# # Simulate the one-variant model without mutation
# # =============================================================================

# res_one = solve_ivp(
#     multi_variant_model_for_loops,
#     (t0, tf),
#     x0_one,
#     t_eval=times,
#     args=(beta_one, gamma_one, sigma_one),
# )

# if not res_one.success:
#     raise RuntimeError(f"One-variant simulation failed: {res_one.message}")

# # One-variant state ordering: [S, R_1, I_1].
# i1_no_mutation = res_one.y[2, :]

# # Cumulative deaths in the no-mutation scenario.
# deaths_no_mutation = res_one.y[1, :] * population * death_rate


# # =============================================================================
# # Discrete distribution of the mutation time
# # =============================================================================
# #
# # On [t_k, t_{k+1}), the mutation hazard is
# #
# #     lambda_k = mutation_rate * I_1(t_k).
# #
# # The probability of a mutation in that interval, conditional on no earlier
# # mutation, is 1 - exp(-lambda_k * dt).
# # =============================================================================

# mutation_times = times[:-1]
# n_mutation_times = len(mutation_times)

# mutation_time_probabilities = np.zeros(n_mutation_times)
# survival_probability = 1.0

# for k in range(n_mutation_times):
#     hazard = mutation_rate * i1_no_mutation[k]

#     mutation_time_probabilities[k] = survival_probability * (
#         1 - np.exp(-hazard * dt)
#     )

#     survival_probability *= np.exp(-hazard * dt)

# probability_no_mutation = survival_probability
# probability_mutation = np.sum(mutation_time_probabilities)

# print(f"Probability of mutation until t = {tf}: {probability_mutation:.4f}")
# print(
#     f"Probability of no mutation until t = {tf}: "
#     f"{probability_no_mutation:.4f}"
# )
# print(
#     f"Total probability mass: "
#     f"{probability_mutation + probability_no_mutation:.8f}"
# )


# # =============================================================================
# # Simulate a trajectory for each possible mutation time
# # =============================================================================
# #
# # Each row represents one possible outcome:
# #
# # - Row 0: no mutation occurs before tf.
# # - Further rows: a mutation occurs at one of the possible mutation times.
# # =============================================================================

# i1_trajectories = []
# i2_trajectories = []
# death_trajectories = []
# trajectory_weights = []

# # Scenario: no mutation.
# i1_trajectories.append(i1_no_mutation.copy())
# i2_trajectories.append(np.zeros_like(times, dtype=float))
# death_trajectories.append(deaths_no_mutation.copy())
# trajectory_weights.append(probability_no_mutation)

# for mutation_index, mutation_time in enumerate(mutation_times):
#     probability = mutation_time_probabilities[mutation_index]

#     # Ignore numerically irrelevant paths.
#     if probability < 1e-14:
#         continue

#     # One-variant state at mutation time: [S, R_1, I_1].
#     x_one_at_mutation = res_one.y[:, mutation_index]

#     if x_one_at_mutation[0] < seed_variant_2:
#         raise ValueError(
#             "The susceptible population at the mutation time is smaller "
#             "than seed_variant_2."
#         )

#     # State ordering for two variants:
#     #
#     # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]
#     #
#     # Cross-immunity against variant 2 induced by variant 1 is represented
#     # by transferring a sigma_12 fraction to compartments R_12 and I_1,12.
#     x0_two = np.zeros(9)

#     x0_two[0] = x_one_at_mutation[0] - seed_variant_2
#     x0_two[1] = (1 - sigma_12) * x_one_at_mutation[1]
#     x0_two[2] = (1 - sigma_12) * x_one_at_mutation[2]
#     x0_two[4] = seed_variant_2
#     x0_two[5] = sigma_12 * x_one_at_mutation[1]
#     x0_two[6] = sigma_12 * x_one_at_mutation[2]

#     assert np.isclose(
#         np.sum(x0_two), 1.0
#     ), "Population is not conserved during the state transformation."

#     # Simulate the two-variant model after mutation.
#     times_after_mutation = times[mutation_index:]

#     res_two = solve_ivp(
#         multi_variant_model_for_loops,
#         (mutation_time, tf),
#         x0_two,
#         t_eval=times_after_mutation,
#         args=(beta_two, gamma_two, sigma_two),
#     )

#     if not res_two.success:
#         raise RuntimeError(
#             f"Two-variant simulation failed at t = {mutation_time}: "
#             f"{res_two.message}"
#         )

#     infectious_two = np.array(
#         [infectious_per_variant(state) for state in res_two.y.T]
#     )

#     # Recovered/protected compartments:
#     # R_1: index 1
#     # R_2: index 3
#     # R_12: index 5
#     recovered_two_total = res_two.y[1, :] + res_two.y[3, :] + res_two.y[5, :]

#     deaths_two = recovered_two_total * population * death_rate

#     # Full trajectories on [t0, tf].
#     # Before mutation, dynamics agree with the one-variant model.
#     i1_path = i1_no_mutation.copy()
#     i2_path = np.zeros_like(times, dtype=float)
#     death_path = deaths_no_mutation.copy()

#     i1_path[mutation_index:] = infectious_two[:, 0]
#     i2_path[mutation_index:] = infectious_two[:, 1]
#     death_path[mutation_index:] = deaths_two

#     i1_trajectories.append(i1_path)
#     i2_trajectories.append(i2_path)
#     death_trajectories.append(death_path)
#     trajectory_weights.append(probability)

# # Convert to arrays.
# i1_trajectories = np.asarray(i1_trajectories)
# i2_trajectories = np.asarray(i2_trajectories)
# death_trajectories = np.asarray(death_trajectories)
# trajectory_weights = np.asarray(trajectory_weights)

# # Correct potential numerical rounding deviations.
# trajectory_weights /= np.sum(trajectory_weights)


# # =============================================================================
# # Expected trajectories
# # =============================================================================

# expected_i1 = np.average(
#     i1_trajectories,
#     axis=0,
#     weights=trajectory_weights,
# )

# expected_i2 = np.average(
#     i2_trajectories,
#     axis=0,
#     weights=trajectory_weights,
# )

# expected_deaths = np.average(
#     death_trajectories,
#     axis=0,
#     weights=trajectory_weights,
# )


# # =============================================================================
# # Pointwise 68% uncertainty intervals
# # =============================================================================

# lower_i1 = np.zeros_like(times, dtype=float)
# upper_i1 = np.zeros_like(times, dtype=float)

# lower_i2 = np.zeros_like(times, dtype=float)
# upper_i2 = np.zeros_like(times, dtype=float)

# lower_deaths = np.zeros_like(times, dtype=float)
# upper_deaths = np.zeros_like(times, dtype=float)

# for time_index in range(len(times)):
#     lower_i1[time_index], upper_i1[time_index] = weighted_quantile(
#         i1_trajectories[:, time_index],
#         quantiles=[0.16, 0.84],
#         weights=trajectory_weights,
#     )

#     lower_i2[time_index], upper_i2[time_index] = weighted_quantile(
#         i2_trajectories[:, time_index],
#         quantiles=[0.16, 0.84],
#         weights=trajectory_weights,
#     )

#     lower_deaths[time_index], upper_deaths[time_index] = weighted_quantile(
#         death_trajectories[:, time_index],
#         quantiles=[0.16, 0.84],
#         weights=trajectory_weights,
#     )


# # =============================================================================
# # Plot: expected dynamics, expected deaths, and 68% uncertainty intervals
# # =============================================================================

# fontsize_text = 16
# fontsize_ticks = 14

# fig = plt.figure()

# # 68% uncertainty interval for I_1.
# plt.fill_between(
#     times,
#     lower_i1 * population,
#     upper_i1 * population,
#     color="#D94A4A",
#     alpha=0.20,
#     linewidth=0,
#     # label=r"68\% interval for $I_1$",
# )

# # 68% uncertainty interval for I_2.
# plt.fill_between(
#     times,
#     lower_i2 * population,
#     upper_i2 * population,
#     color="#F28E2B",
#     alpha=0.20,
#     linewidth=0,
#     # label=r"68\% interval for $I_2$",
# )

# # 68% uncertainty interval for cumulative deaths.
# plt.fill_between(
#     times,
#     lower_deaths,
#     upper_deaths,
#     color="#000000",
#     alpha=0.15,
#     linewidth=0,
#     # label="68% interval for deaths",
# )

# # Expected infectious persons with variant 1.
# plt.plot(
#     times,
#     expected_i1 * population,
#     color="#D94A4A",
#     linewidth=2.5,
#     label=r"Expected $I_1$",
# )

# # Expected infectious persons with variant 2.
# plt.plot(
#     times,
#     expected_i2 * population,
#     color="#F28E2B",
#     linewidth=2.5,
#     label=r"Expected $I_2$",
# )

# # Expected cumulative deaths.
# plt.plot(
#     times,
#     expected_deaths,
#     color="#000000",
#     linewidth=2.5,
#     label="Expected deaths",
# )

# plt.xlabel("Time", fontsize=fontsize_text)
# plt.ylabel("Compartment size", fontsize=fontsize_text)

# plt.xlim(t0, tf)
# plt.ylim(bottom=0, top=15)

# plt.legend(
#     fontsize=fontsize_ticks,
#     ncol=1,
#     loc="upper left",
# )

# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.tight_layout()
# plt.show()


"""
Expected epidemic dynamics for a two-variant model with a stochastic
mutation time, including expected cumulative deaths and pointwise
68% uncertainty intervals.

Scenario:
- 12 risk-group persons are vaccinated initially.
- Vaccinated persons are assumed to be fully protected against variant 1.
- Mutation intensity is proportional to the number of individuals infectious
  with variant 1:

      lambda(t) = mutation_rate * I_1(t)

- Deaths are approximated by a fixed mortality rate among unvaccinated persons:

      D(t) = death_rate * cumulative infections(t)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    infectious_per_variant,
    multi_variant_model_for_loops,
)


def weighted_quantile(values, quantiles, weights):
    """Compute weighted empirical quantiles.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional array of values.
    quantiles : float or array-like
        Quantiles in [0, 1].
    weights : np.ndarray
        Non-negative probability weights associated with values.

    Returns
    -------
    np.ndarray
        Weighted empirical quantiles.
    """
    values = np.asarray(values)
    quantiles = np.atleast_1d(quantiles)
    weights = np.asarray(weights)

    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length.")

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("The sum of weights must be positive.")

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]

    cumulative_weights = np.cumsum(weights_sorted)
    cumulative_weights /= cumulative_weights[-1]

    indices = np.searchsorted(cumulative_weights, quantiles, side="left")
    indices = np.clip(indices, 0, len(values_sorted) - 1)

    return values_sorted[indices]


# =============================================================================
# Parameters
# =============================================================================

t0 = 0
tf = 200
dt = 1.0

population = 80

# Epidemiological parameters before mutation.
beta_one = [0.15]
gamma_one = [0.1]
sigma_one = np.array([[1.0]])

# Epidemiological parameters after the appearance of variant 2.
beta_two = [0.15, 0.18]
gamma_two = [0.1, 0.1]

sigma_12 = 0.5
sigma_two = np.array(
    [
        [1.0, sigma_12],
        [sigma_12, 1.0],
    ]
)

# Mutation intensity:
# lambda(t) = mutation_rate * I_1(t)
mutation_rate = 0.5

# A mutation introduces one individual infectious with variant 2.
seed_variant_2 = 1 / population

# Time grid.
times = np.arange(t0, tf + dt, dt)


# =============================================================================
# Vaccination and mortality scenario
# =============================================================================

risk_group_persons = 25
other_persons = 55

vaccinated_risk_persons = 0

# One initially recovered person and one initially infectious person are kept
# as in the previous example.
initial_recovered = 1
initial_infectious = 1

# Vaccinated persons are represented as initially protected individuals.
initial_protected = vaccinated_risk_persons + initial_recovered

# State ordering for the one-variant model:
# [S, R_1, I_1]
x0_one = np.array(
    [
        (
            population
            - vaccinated_risk_persons
            - initial_recovered
            - initial_infectious
        )
        / population,
        initial_protected / population,
        initial_infectious / population,
    ]
)

assert np.isclose(
    np.sum(x0_one), 1.0
), "The initial population fractions do not sum to one."

# Mortality rates:
# - Risk-group persons: 0.5
# - Other persons:      0.05
#
# Vaccinated risk-group persons are assumed not to die from infection.
# The average death rate is therefore calculated over the unvaccinated
# population only.
remaining_risk_group_persons = risk_group_persons - vaccinated_risk_persons

death_rate = (0.3 * remaining_risk_group_persons + 0.15 * other_persons) / (
    population - vaccinated_risk_persons
)

print(f"Vaccinated risk-group persons: {vaccinated_risk_persons}")
print(f"Remaining risk-group persons: {remaining_risk_group_persons}")
print(f"Average death rate among unvaccinated persons: {death_rate:.4f}")


# =============================================================================
# Simulate the one-variant model without mutation
# =============================================================================

res_one = solve_ivp(
    multi_variant_model_for_loops,
    (t0, tf),
    x0_one,
    t_eval=times,
    args=(beta_one, gamma_one, sigma_one),
)

if not res_one.success:
    raise RuntimeError(f"One-variant simulation failed: {res_one.message}")

# State ordering: [S, R_1, I_1].
i1_no_mutation = res_one.y[2, :]

# R_1 also contains the initially vaccinated and initially recovered persons.
# These should not be counted as persons infected during the simulation.
recovered_from_infection_no_mutation = np.maximum(
    res_one.y[1, :] - initial_protected / population,
    0.0,
)

deaths_no_mutation = (
    recovered_from_infection_no_mutation * population * death_rate
)


# =============================================================================
# Discrete distribution of the mutation time
# =============================================================================
#
# On [t_k, t_{k+1}), the mutation hazard is
#
#     lambda_k = mutation_rate * I_1(t_k).
#
# If no mutation occurred before t_k, then the probability of a mutation in
# [t_k, t_{k+1}) is
#
#     1 - exp(-lambda_k * dt).
# =============================================================================

mutation_times = times[:-1]
n_mutation_times = len(mutation_times)

mutation_time_probabilities = np.zeros(n_mutation_times)
survival_probability = 1.0

for k in range(n_mutation_times):
    hazard = mutation_rate * i1_no_mutation[k]

    mutation_time_probabilities[k] = survival_probability * (
        1 - np.exp(-hazard * dt)
    )

    survival_probability *= np.exp(-hazard * dt)

probability_no_mutation = survival_probability
probability_mutation = np.sum(mutation_time_probabilities)

print(f"Probability of mutation until t = {tf}: {probability_mutation:.4f}")
print(
    f"Probability of no mutation until t = {tf}: "
    f"{probability_no_mutation:.4f}"
)
print(
    f"Total probability mass: "
    f"{probability_mutation + probability_no_mutation:.8f}"
)


# =============================================================================
# Simulate one trajectory for every possible mutation time
# =============================================================================
#
# The first trajectory corresponds to no mutation.
# Every further trajectory corresponds to a mutation at a particular time.
# =============================================================================

i1_trajectories = []
i2_trajectories = []
death_trajectories = []
trajectory_weights = []

# Scenario 0: no mutation.
i1_trajectories.append(i1_no_mutation.copy())
i2_trajectories.append(np.zeros_like(times, dtype=float))
death_trajectories.append(deaths_no_mutation.copy())
trajectory_weights.append(probability_no_mutation)

for mutation_index, mutation_time in enumerate(mutation_times):
    probability = mutation_time_probabilities[mutation_index]

    # Ignore numerically irrelevant trajectories.
    if probability < 1e-14:
        continue

    # State ordering of the one-variant model: [S, R_1, I_1].
    x_one_at_mutation = res_one.y[:, mutation_index]

    if x_one_at_mutation[0] < seed_variant_2:
        raise ValueError(
            "The susceptible population at the mutation time is smaller "
            "than seed_variant_2."
        )

    # State ordering for the two-variant model:
    #
    # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]
    #
    # A sigma_12 fraction of persons protected against variant 1 is also
    # protected against variant 2. These are transferred to R_12 or I_1,12.
    x0_two = np.zeros(9)

    x0_two[0] = x_one_at_mutation[0] - seed_variant_2
    x0_two[1] = (1 - sigma_12) * x_one_at_mutation[1]
    x0_two[2] = (1 - sigma_12) * x_one_at_mutation[2]
    x0_two[4] = seed_variant_2
    x0_two[5] = sigma_12 * x_one_at_mutation[1]
    x0_two[6] = sigma_12 * x_one_at_mutation[2]

    assert np.isclose(
        np.sum(x0_two), 1.0
    ), "Population is not conserved during the state transformation."

    # Simulate the two-variant system after the mutation.
    times_after_mutation = times[mutation_index:]

    res_two = solve_ivp(
        multi_variant_model_for_loops,
        (mutation_time, tf),
        x0_two,
        t_eval=times_after_mutation,
        args=(beta_two, gamma_two, sigma_two),
    )

    if not res_two.success:
        raise RuntimeError(
            f"Two-variant simulation failed at t = {mutation_time}: "
            f"{res_two.message}"
        )

    infectious_two = np.array(
        [infectious_per_variant(state) for state in res_two.y.T]
    )

    # Recovered/protected compartments:
    # R_1:  index 1
    # R_2:  index 3
    # R_12: index 5
    recovered_two_total = res_two.y[1, :] + res_two.y[3, :] + res_two.y[5, :]

    # Subtract the initial vaccinated/protected fraction, which is contained in
    # the recovered/protected compartments but does not represent infection.
    recovered_from_infection_two = np.maximum(
        recovered_two_total - initial_protected / population,
        0.0,
    )

    deaths_two = recovered_from_infection_two * population * death_rate

    # Construct trajectories on the full time interval [t0, tf].
    # Before the mutation, only variant 1 is present.
    i1_path = i1_no_mutation.copy()
    i2_path = np.zeros_like(times, dtype=float)
    death_path = deaths_no_mutation.copy()

    i1_path[mutation_index:] = infectious_two[:, 0]
    i2_path[mutation_index:] = infectious_two[:, 1]
    death_path[mutation_index:] = deaths_two

    i1_trajectories.append(i1_path)
    i2_trajectories.append(i2_path)
    death_trajectories.append(death_path)
    trajectory_weights.append(probability)

# Convert lists to arrays.
i1_trajectories = np.asarray(i1_trajectories)
i2_trajectories = np.asarray(i2_trajectories)
death_trajectories = np.asarray(death_trajectories)
trajectory_weights = np.asarray(trajectory_weights)

# Correct possible numerical rounding errors.
trajectory_weights /= np.sum(trajectory_weights)


# =============================================================================
# Expected trajectories
# =============================================================================

expected_i1 = np.average(
    i1_trajectories,
    axis=0,
    weights=trajectory_weights,
)

expected_i2 = np.average(
    i2_trajectories,
    axis=0,
    weights=trajectory_weights,
)

expected_deaths = np.average(
    death_trajectories,
    axis=0,
    weights=trajectory_weights,
)


# =============================================================================
# Pointwise 68% uncertainty intervals
# =============================================================================

lower_i1 = np.zeros_like(times, dtype=float)
upper_i1 = np.zeros_like(times, dtype=float)

lower_i2 = np.zeros_like(times, dtype=float)
upper_i2 = np.zeros_like(times, dtype=float)

lower_deaths = np.zeros_like(times, dtype=float)
upper_deaths = np.zeros_like(times, dtype=float)

for time_index in range(len(times)):
    lower_i1[time_index], upper_i1[time_index] = weighted_quantile(
        i1_trajectories[:, time_index],
        quantiles=[0.16, 0.84],
        weights=trajectory_weights,
    )

    lower_i2[time_index], upper_i2[time_index] = weighted_quantile(
        i2_trajectories[:, time_index],
        quantiles=[0.16, 0.84],
        weights=trajectory_weights,
    )

    lower_deaths[time_index], upper_deaths[time_index] = weighted_quantile(
        death_trajectories[:, time_index],
        quantiles=[0.16, 0.84],
        weights=trajectory_weights,
    )


# =============================================================================
# Plot: expected dynamics, deaths, and pointwise 68% uncertainty intervals
# =============================================================================

fontsize_text = 16
fontsize_ticks = 14

fig = plt.figure()

# Uncertainty interval for I_1.
plt.fill_between(
    times,
    lower_i1 * population,
    upper_i1 * population,
    color="#D94A4A",
    alpha=0.20,
    linewidth=0,
    # label=r"68\% interval for $I_1$",
)

# Uncertainty interval for I_2.
plt.fill_between(
    times,
    lower_i2 * population,
    upper_i2 * population,
    color="#F28E2B",
    alpha=0.20,
    linewidth=0,
    # label=r"68\% interval for $I_2$",
)

# Uncertainty interval for cumulative deaths.
plt.fill_between(
    times,
    lower_deaths,
    upper_deaths,
    color="#000000",
    alpha=0.15,
    linewidth=0,
    # label="68% interval for deaths",
)

# Expected infectious population for variant 1.
plt.plot(
    times,
    expected_i1 * population,
    color="#D94A4A",
    linewidth=2.5,
    label=r"Expected $I_1$",
)

# Expected infectious population for variant 2.
plt.plot(
    times,
    expected_i2 * population,
    color="#F28E2B",
    linewidth=2.5,
    label=r"Expected $I_2$",
)

# Expected cumulative deaths.
plt.plot(
    times,
    expected_deaths,
    color="#000000",
    linewidth=2.5,
    label="Expected deaths",
)

plt.xlabel("Time", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)

plt.xlim(t0, tf)
plt.ylim(bottom=0, top=15)

plt.legend(
    fontsize=fontsize_ticks,
    ncol=1,
    loc="upper left",
)

plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)

plt.tight_layout()
plt.show()


# """
# Expected final deaths for different numbers of vaccinated persons.

# A second variant can emerge at a stochastic mutation time with intensity

#     lambda(t) = mutation_rate * I_1(t).

# For each vaccination level, the expected final number of deaths and its
# pointwise 68% uncertainty interval over mutation times are computed.
# """

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.integrate import solve_ivp

# from cross_immunity_model import multi_variant_model_for_loops


# def weighted_quantile(values, quantiles, weights):
#     """Compute weighted empirical quantiles.

#     Parameters
#     ----------
#     values : np.ndarray
#         One-dimensional array of values.
#     quantiles : float or array-like
#         Quantiles in [0, 1].
#     weights : np.ndarray
#         Non-negative probability weights associated with values.

#     Returns
#     -------
#     np.ndarray
#         Weighted empirical quantiles.
#     """
#     values = np.asarray(values)
#     quantiles = np.atleast_1d(quantiles)
#     weights = np.asarray(weights)

#     if len(values) != len(weights):
#         raise ValueError("values and weights must have equal length.")

#     if np.any(weights < 0):
#         raise ValueError("weights must be non-negative.")

#     if np.sum(weights) <= 0:
#         raise ValueError("The sum of weights must be positive.")

#     order = np.argsort(values)
#     values_sorted = values[order]
#     weights_sorted = weights[order]

#     cumulative_weights = np.cumsum(weights_sorted)
#     cumulative_weights /= cumulative_weights[-1]

#     indices = np.searchsorted(cumulative_weights, quantiles, side="left")
#     indices = np.clip(indices, 0, len(values_sorted) - 1)

#     return values_sorted[indices]


# # =============================================================================
# # Model parameters
# # =============================================================================

# t0 = 0
# tf = 200
# dt = 1.0

# population = 80
# times = np.arange(t0, tf + dt, dt)
# mutation_times = times[:-1]

# # One-variant model before mutation.
# beta_one = [0.15]
# gamma_one = [0.1]
# sigma_one = np.array([[1.0]])

# # Two-variant model after mutation.
# beta_two = [0.15, 0.18]
# gamma_two = [0.1, 0.1]

# sigma_12 = 0.5
# sigma_two = np.array(
#     [
#         [1.0, sigma_12],
#         [sigma_12, 1.0],
#     ]
# )

# # Mutation intensity:
# # lambda(t) = mutation_rate * I_1(t)
# mutation_rate = 0.5

# # A mutation introduces one individual infectious with variant 2.
# seed_variant_2 = 1 / population

# # Population structure.
# risk_group_persons = 25
# other_persons = 55

# # Mortality probabilities after infection.
# death_rate_risk_group = 0.3
# death_rate_other_persons = 0.15

# # Initial epidemic state:
# # one recovered person and one infectious person.
# initial_recovered = 1
# initial_infectious = 1


# # =============================================================================
# # Expected deaths for one vaccination level
# # =============================================================================


# def expected_final_deaths_with_vaccination(vaccinated_persons: int):
#     """Compute expected final deaths and the central 68% interval.

#     Vaccination is assigned first to risk-group persons and subsequently to
#     other persons. Vaccinated persons are assumed to be fully protected
#     against the first variant.

#     Parameters
#     ----------
#     vaccinated_persons : int
#         Number of vaccinated persons.

#     Returns
#     -------
#     expected_final_deaths : float
#         Expected cumulative deaths at time tf.
#     lower_final_deaths : float
#         Lower limit of the central 68% uncertainty interval.
#     upper_final_deaths : float
#         Upper limit of the central 68% uncertainty interval.
#     """
#     maximum_vaccinated = population - initial_recovered - initial_infectious

#     if not 0 <= vaccinated_persons <= maximum_vaccinated:
#         raise ValueError(
#             f"vaccinated_persons must be between 0 and "
#             f"{maximum_vaccinated}."
#         )

#     # -------------------------------------------------------------------------
#     # Vaccination scenario and average death rate
#     # -------------------------------------------------------------------------

#     # Vaccinate risk-group persons first.
#     vaccinated_risk_persons = min(
#         vaccinated_persons,
#         risk_group_persons,
#     )
#     vaccinated_other_persons = max(
#         vaccinated_persons - risk_group_persons,
#         0,
#     )

#     remaining_risk_persons = risk_group_persons - vaccinated_risk_persons
#     remaining_other_persons = other_persons - vaccinated_other_persons

#     # The average death probability among the unvaccinated population.
#     unvaccinated_persons = population - vaccinated_persons

#     death_rate = (
#         death_rate_risk_group * remaining_risk_persons
#         + death_rate_other_persons * remaining_other_persons
#     ) / unvaccinated_persons

#     # Vaccinated persons and the initially recovered person are represented
#     # as initially protected persons in R_1.
#     initial_protected = vaccinated_persons + initial_recovered

#     # State ordering for the one-variant model:
#     # [S, R_1, I_1]
#     x0_one = np.array(
#         [
#             (
#                 population
#                 - vaccinated_persons
#                 - initial_recovered
#                 - initial_infectious
#             )
#             / population,
#             initial_protected / population,
#             initial_infectious / population,
#         ]
#     )

#     assert np.isclose(
#         np.sum(x0_one), 1.0
#     ), "Initial population fractions do not sum to one."

#     # -------------------------------------------------------------------------
#     # One-variant trajectory without mutation
#     # -------------------------------------------------------------------------

#     res_one = solve_ivp(
#         multi_variant_model_for_loops,
#         (t0, tf),
#         x0_one,
#         t_eval=times,
#         args=(beta_one, gamma_one, sigma_one),
#     )

#     if not res_one.success:
#         raise RuntimeError(
#             f"One-variant simulation failed for "
#             f"{vaccinated_persons} vaccinated persons."
#         )

#     # One-variant state ordering: [S, R_1, I_1].
#     i1_no_mutation = res_one.y[2, :]

#     # R_1 contains initially protected persons and persons who recovered
#     # from infection. Only the latter are counted as infections/deaths.
#     recovered_from_infection_no_mutation = np.maximum(
#         res_one.y[1, :] - initial_protected / population,
#         0.0,
#     )

#     deaths_no_mutation_final = (
#         recovered_from_infection_no_mutation[-1] * population * death_rate
#     )

#     # -------------------------------------------------------------------------
#     # Discrete distribution of the mutation time
#     # -------------------------------------------------------------------------

#     mutation_time_probabilities = np.zeros(len(mutation_times))
#     survival_probability = 1.0

#     for mutation_index in range(len(mutation_times)):
#         hazard = mutation_rate * i1_no_mutation[mutation_index]

#         mutation_time_probabilities[mutation_index] = survival_probability * (
#             1 - np.exp(-hazard * dt)
#         )

#         survival_probability *= np.exp(-hazard * dt)

#     probability_no_mutation = survival_probability

#     # Store final deaths for all possible outcomes:
#     # - no mutation;
#     # - mutation at each possible mutation time.
#     final_death_values = [deaths_no_mutation_final]
#     final_death_weights = [probability_no_mutation]

#     # -------------------------------------------------------------------------
#     # Two-variant trajectories after mutation
#     # -------------------------------------------------------------------------

#     for mutation_index, mutation_time in enumerate(mutation_times):
#         probability = mutation_time_probabilities[mutation_index]

#         if probability < 1e-14:
#             continue

#         # One-variant state at mutation time:
#         # [S, R_1, I_1]
#         x_one_at_mutation = res_one.y[:, mutation_index]

#         # If fewer than one susceptible person remains, the new variant
#         # cannot be seeded under this deterministic construction.
#         if x_one_at_mutation[0] < seed_variant_2:
#             final_death_values.append(deaths_no_mutation_final)
#             final_death_weights.append(probability)
#             continue

#         # State ordering for two variants:
#         #
#         # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]
#         #
#         # A fraction sigma_12 of R_1 and I_1 obtains protection against
#         # variant 2 and is therefore transferred to R_12 and I_1,12.
#         x0_two = np.zeros(9)

#         x0_two[0] = x_one_at_mutation[0] - seed_variant_2
#         x0_two[1] = (1 - sigma_12) * x_one_at_mutation[1]
#         x0_two[2] = (1 - sigma_12) * x_one_at_mutation[2]
#         x0_two[4] = seed_variant_2
#         x0_two[5] = sigma_12 * x_one_at_mutation[1]
#         x0_two[6] = sigma_12 * x_one_at_mutation[2]

#         assert np.isclose(
#             np.sum(x0_two), 1.0
#         ), "Population is not conserved during the state transformation."

#         res_two = solve_ivp(
#             multi_variant_model_for_loops,
#             (mutation_time, tf),
#             x0_two,
#             t_eval=times[mutation_index:],
#             args=(beta_two, gamma_two, sigma_two),
#         )

#         if not res_two.success:
#             raise RuntimeError(
#                 f"Two-variant simulation failed for "
#                 f"{vaccinated_persons} vaccinated persons at "
#                 f"t = {mutation_time}."
#             )

#         # Recovered/protected compartments in the two-variant model:
#         # R_1: index 1, R_2: index 3, R_12: index 5.
#         recovered_two_total_final = (
#             res_two.y[1, -1] + res_two.y[3, -1] + res_two.y[5, -1]
#         )

#         # Do not count initially vaccinated/protected persons as infections.
#         recovered_from_infection_two_final = max(
#             recovered_two_total_final - initial_protected / population,
#             0.0,
#         )

#         deaths_two_final = (
#             recovered_from_infection_two_final * population * death_rate
#         )

#         final_death_values.append(deaths_two_final)
#         final_death_weights.append(probability)

#     # -------------------------------------------------------------------------
#     # Expected deaths and central 68% uncertainty interval
#     # -------------------------------------------------------------------------

#     final_death_values = np.asarray(final_death_values)
#     final_death_weights = np.asarray(final_death_weights)

#     # Correct small floating-point deviations from total probability one.
#     final_death_weights /= np.sum(final_death_weights)

#     expected_final_deaths = np.average(
#         final_death_values,
#         weights=final_death_weights,
#     )

#     lower_final_deaths, upper_final_deaths = weighted_quantile(
#         final_death_values,
#         quantiles=[0.16, 0.84],
#         weights=final_death_weights,
#     )

#     return (
#         expected_final_deaths,
#         lower_final_deaths,
#         upper_final_deaths,
#     )


# # =============================================================================
# # Evaluate vaccination strategies
# # =============================================================================

# maximum_vaccinated = population - initial_recovered - initial_infectious

# vaccinated_persons_values = np.arange(
#     0,
#     maximum_vaccinated + 1,
# )

# expected_final_deaths_values = np.zeros(len(vaccinated_persons_values))
# lower_final_deaths = np.zeros(len(vaccinated_persons_values))
# upper_final_deaths = np.zeros(len(vaccinated_persons_values))

# for index, vaccinated_persons in enumerate(vaccinated_persons_values):
#     (
#         expected_final_deaths_values[index],
#         lower_final_deaths[index],
#         upper_final_deaths[index],
#     ) = expected_final_deaths_with_vaccination(vaccinated_persons)

#     print(
#         f"Vaccinated persons: {vaccinated_persons:2d} | "
#         f"Expected deaths: {expected_final_deaths_values[index]:.4f} | "
#         f"68% interval: "
#         f"[{lower_final_deaths[index]:.4f}, "
#         f"{upper_final_deaths[index]:.4f}]"
#     )


# # =============================================================================
# # Plot
# # =============================================================================

# fontsize_text = 16
# fontsize_ticks = 14

# fig = plt.figure()

# # Central 68% uncertainty interval.
# # No label is intentionally provided, so it is not shown in the legend.
# plt.fill_between(
#     vaccinated_persons_values,
#     lower_final_deaths,
#     upper_final_deaths,
#     color="#000000",
#     alpha=0.15,
#     linewidth=0,
# )

# # Expected number of deaths.
# plt.plot(
#     vaccinated_persons_values,
#     expected_final_deaths_values,
#     color="#000000",
#     linewidth=2.5,
#     label="Expected deaths",
# )

# # The first 25 vaccinated persons belong to the risk group.
# plt.axvline(
#     x=risk_group_persons,
#     color="#7F7F7F",
#     linestyle="--",
#     linewidth=2,
#     label="Risk-group persons",
# )

# plt.xlabel("Vaccinated persons", fontsize=fontsize_text)
# plt.ylabel("Expected number of deaths", fontsize=fontsize_text)

# plt.xlim(0, maximum_vaccinated)
# plt.ylim(bottom=0)

# plt.legend(fontsize=fontsize_ticks)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.tight_layout()
# plt.show()
