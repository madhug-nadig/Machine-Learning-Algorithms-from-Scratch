import copy

import numpy as np
import pandas as pd


class CustomGeneticAlgorithm():

    def server_present(self, server, time):
        server_start_time = server[1]
        server_duration = server[2]
        server_end_time = server_start_time + server_duration
        if (time >= server_start_time) and (time < server_end_time):
            return True
        return False

    def deployed_to_hourlyplanning(self, deployed_hourly_cron_capacity):

        deployed_hourly_cron_capacity_week = []
        for day in deployed_hourly_cron_capacity:

            deployed_hourly_cron_capacity_day = []
            for server in day:

                server_present_hour = []
                for time in range(0, 24):

                    server_present_hour.append(
                        self.server_present(server, time))

                deployed_hourly_cron_capacity_day.append(server_present_hour)

            deployed_hourly_cron_capacity_week.append(
                deployed_hourly_cron_capacity_day)

        deployed_hourly_cron_capacity_week = np.array(
            deployed_hourly_cron_capacity_week).sum(axis=1)
        return deployed_hourly_cron_capacity_week

    def generate_random_plan(self, n_days, n_racks):
        period_planning = []
        for _ in range(n_days):
            day_planning = []
            for server_id in range(n_racks):
                start_time = np.random.randint(0, 23)
                machines = np.random.randint(0, 12)
                server = [server_id, start_time, machines]
                day_planning.append(server)

            period_planning.append(day_planning)

        return period_planning

    def generate_initial_population(self, population_size, n_days=7, n_racks=11):
        population = []
        for _ in range(population_size):
            member = self.generate_random_plan(
                n_days=n_days, n_racks=n_racks)
            population.append(member)
        return population

    def calculate_fitness(self, deployed_hourly_cron_capacity, required_hourly_cron_capacity):
        deviation = deployed_hourly_cron_capacity - required_hourly_cron_capacity
        overcapacity = abs(deviation[deviation > 0].sum())
        undercapacity = abs(deviation[deviation < 0].sum())

        overcapacity_cost = 0.5
        undercapacity_cost = 3

        fitness = overcapacity_cost * overcapacity + undercapacity_cost * undercapacity
        return fitness

    def crossover(self, population, n_offspring):
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

    def mutate_parent(self, parent, n_mutations):
        size1 = parent.shape[0]
        size2 = parent.shape[1]

        for _ in range(n_mutations):
            rand1 = np.random.randint(0, size1)
            rand2 = np.random.randint(0, size2)
            rand3 = np.random.randint(0, 2)
            parent[rand1, rand2, rand3] = np.random.randint(0, 12)
        return parent

    def mutate_gen(self, population, n_mutations):
        mutated_population = []
        for parent in population:
            mutated_population.append(self.mutate_parent(parent, n_mutations))
        return mutated_population

    def is_acceptable(self, parent):
        return np.logical_not((np.array(parent)[:, :, 2:] > 12).any())

    def select_acceptable(self, population):
        population = [
            parent for parent in population if self.is_acceptable(parent)]
        return population

    def select_best(self, population, required_hourly_cron_capacity, n_best):
        fitness = []
        for idx, deployed_hourly_cron_capacity in enumerate(population):

            deployed_hourly_cron_capacity = self.deployed_to_hourlyplanning(
                deployed_hourly_cron_capacity)
            parent_fitness = self.calculate_fitness(deployed_hourly_cron_capacity,
                                                    required_hourly_cron_capacity)
            fitness.append([idx, parent_fitness])

        print('Current generation\'s optimal schedule has cost: {}'.format(
            pd.DataFrame(fitness)[1].min()))

        fitness_tmp = pd.DataFrame(fitness).sort_values(
            by=1, ascending=True).reset_index(drop=True)
        selected_parents_idx = list(fitness_tmp.iloc[:n_best, 0])
        selected_parents = [parent for idx, parent in enumerate(
            population) if idx in selected_parents_idx]

        return selected_parents

    def run(self, required_hourly_cron_capacity, n_iterations, n_population_size=500):

        population = self.generate_initial_population(
            population_size=n_population_size, n_days=5, n_racks=24)
        for _ in range(n_iterations):
            population = self.select_acceptable(population)
            population = self.select_best(
                population, required_hourly_cron_capacity, n_best=100)
            population = self.crossover(
                population, n_offspring=n_population_size)
            population = self.mutate_gen(population, n_mutations=1)

        best_child = self.select_best(
            population, required_hourly_cron_capacity, n_best=1)
        return best_child


def main():

    # Reading from the data file
    df = pd.read_csv("./data/cron_jobs_schedule.csv")

    dataset = df.astype(int).values.tolist()

    required_hourly_cron_capacity = [
        [0 for _ in range(24)] for _ in range(5)]

    for record in dataset:
        required_hourly_cron_capacity[record[1]][record[2]] += record[3]

    genetic_algorithm = CustomGeneticAlgorithm()
    optimal_schedule = genetic_algorithm.run(
        required_hourly_cron_capacity, n_iterations=100)
    print('\nOptimal Server Schedule: \n',
          genetic_algorithm.deployed_to_hourlyplanning(optimal_schedule[0]))


if __name__ == "__main__":
    main()
