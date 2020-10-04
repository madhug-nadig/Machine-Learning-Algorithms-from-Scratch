import copy

import numpy as np
import pandas as pd

required_hourly_cron_capacity = [
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 2, 2, 2, 6, 6, 6, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 2, 2, 2, 6, 6, 6, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 2, 2, 2, 6, 6, 6, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 2, 2, 2, 6, 6, 6, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 2, 2, 2, 6, 6, 6, 2, 2, 2, 2]
]


def server_present(server, time):
    server_start_time = server[1]
    server_duration = server[2]
    server_end_time = server_start_time + server_duration
    if (time >= server_start_time) and (time < server_end_time):
        return True
    return False


def deployed_to_hourlyplanning(deployed_hourly_cron_capacity):

    deployed_hourly_cron_capacity_week = []
    for day in deployed_hourly_cron_capacity:

        deployed_hourly_cron_capacity_day = []
        for server in day:

            server_present_hour = []
            for time in range(0, 24):

                server_present_hour.append(server_present(server, time))

            deployed_hourly_cron_capacity_day.append(server_present_hour)

        deployed_hourly_cron_capacity_week.append(
            deployed_hourly_cron_capacity_day)

    deployed_hourly_cron_capacity_week = np.array(
        deployed_hourly_cron_capacity_week).sum(axis=1)
    return deployed_hourly_cron_capacity_week


def generate_random_plan(n_days, n_servers):
    period_planning = []
    for _ in range(n_days):
        day_planning = []
        for server_id in range(n_servers):
            start_time = np.random.randint(0, 23)
            duration = np.random.randint(0, 10)
            server = [server_id, start_time, duration]
            day_planning.append(server)

        period_planning.append(day_planning)

    return period_planning


def generate_initial_population(population_size, n_days=7, n_servers=11):
    population = []
    for _ in range(population_size):
        member = generate_random_plan(n_days=n_days, n_servers=n_servers)
        population.append(member)
    return population


def calculate_fitness(deployed_hourly_cron_capacity, required_hourly_cron_capacity):
    deviation = deployed_hourly_cron_capacity - required_hourly_cron_capacity
    overcapacity = abs(deviation[deviation > 0].sum())
    undercapacity = abs(deviation[deviation < 0].sum())

    overcapacity_cost = 1
    undercapacity_cost = 1

    fitness = overcapacity_cost * overcapacity + undercapacity_cost * undercapacity
    return fitness


def crossover(population, n_offspring):
    n_population = len(population)

    offspring = []

    for _ in range(n_offspring):
        random_one = population[np.random.randint(
            low=0, high=n_population - 1)]
        random_two = population[np.random.randint(
            low=0, high=n_population - 1)]

        dad_mask = np.random.randint(0, 2, size=np.array(random_one).shape)
        mom_mask = np.logical_not(dad_mask)

        child = np.add(np.multiply(random_one, dad_mask),
                       np.multiply(random_two, mom_mask))

        offspring.append(child)
    return offspring


def mutate_parent(parent, n_mutations):
    size1 = parent.shape[0]
    size2 = parent.shape[1]

    for _ in range(n_mutations):
        rand1 = np.random.randint(0, size1)
        rand2 = np.random.randint(0, size2)
        rand3 = np.random.randint(0, 2)
        parent[rand1, rand2, rand3] = np.random.randint(0, 10)
    return parent


def mutate_gen(population, n_mutations):
    mutated_population = []
    for parent in population:
        mutated_population.append(mutate_parent(parent, n_mutations))
    return mutated_population


def is_acceptable(parent):
    return np.logical_not((np.array(parent)[:, :, 2:] > 10).any())


def select_acceptable(population):
    population = [parent for parent in population if is_acceptable(parent)]
    return population


def select_best(population, required_hourly_cron_capacity, n_best):
    fitness = []
    for idx, deployed_hourly_cron_capacity in enumerate(population):

        deployed_hourly_cron_capacity = deployed_to_hourlyplanning(
            deployed_hourly_cron_capacity)
        parent_fitness = calculate_fitness(deployed_hourly_cron_capacity,
                                           required_hourly_cron_capacity)
        fitness.append([idx, parent_fitness])

    print('generations best is: {}'.format(
        pd.DataFrame(fitness)[1].min()))

    fitness_tmp = pd.DataFrame(fitness).sort_values(
        by=1, ascending=True).reset_index(drop=True)
    selected_parents_idx = list(fitness_tmp.iloc[:n_best, 0])
    selected_parents = [parent for idx, parent in enumerate(
        population) if idx in selected_parents_idx]

    return selected_parents


def genetic_algo(required_hourly_cron_capacity, n_iterations):

    population_size = 500

    population = generate_initial_population(
        population_size=population_size, n_days=5, n_servers=11)
    for _ in range(n_iterations):
        population = select_acceptable(population)
        population = select_best(
            population, required_hourly_cron_capacity, n_best=100)
        population = crossover(population, n_offspring=population_size)
        population = mutate_gen(population, n_mutations=1)

    best_child = select_best(
        population, required_hourly_cron_capacity, n_best=1)
    return best_child


best_planning = genetic_algo(required_hourly_cron_capacity, n_iterations=100)
