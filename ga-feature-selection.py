from pathlib import Path
from random import randint, choices, random
import csv
import numpy
import sys

NUM_GEN = 100
NUM_FEAT = 30
POP_SIZE = 100
ELITISM = 0.1
MUTATION_CHANCE = 0.1
ALPHA = 0.1


def main():
    data = read_file('./wbcd.data', './wbcd.names')
    # solution = run_feature_selection(data)

    # Setup up initial population and variables
    # individual_len = info[0]
    # capacity = info[1]

    # vs = [x[0] for x in data]
    # total = 0
    # if solution is not None:
    #     for i, value in enumerate(vs):
    #         if solution[i] == 1:
    #             total += value

    # print(total)


def run_feature_selection(data):
    pop = [[randint(0, 1) for __ in range(NUM_FEAT)]
           for ___ in range(POP_SIZE)]
    pop_fit = [None] * POP_SIZE

    # INITIALISE POPULATION

    # values = [row[0] for row in data]
    # weights = [row[1] for row in data]

    best_feasible = None
    best_fit = 0

    for __ in NUM_GEN:
        # Calc fitness of current pop
        for i, ind in enumerate(pop):
            pop_fit[i] = calc_feature_selection(ind, values, weights, capacity)
        # Find new best feasible
        best_ind = max(range(len(pop_fit)), key=pop_fit.__getitem__)
        if pop_fit[best_ind] > best_fit:
            best_feasible = pop[best_ind]
            best_fit = pop_fit[best_ind]

        pop = create_new_pop(pop, pop_fit, individual_len)

    return best_feasible


def calc_feature_selection():
    print()


# Deep copy arrays and then do elitism.
def elitism_selection(pop, fits):
    clone = pop.copy()
    clone_fit = fits.copy()
    elitists = []

    for __ in range(round(ELITISM * POP_SIZE)):
        best_ind = max(range(len(clone_fit)), key=clone_fit.__getitem__)
        elitists.append(clone[best_ind])
        del clone[best_ind]
        del clone_fit[best_ind]

    return elitists


def roulette_selection(prev_pop, prev_fits, k):
    total_fitness = sum(prev_fits)
    fitness_proportions = [x / total_fitness for x in prev_fits]
    # Uses the proportional weighting to select TWO parents
    pop = []
    try:
        pop = choices(population=prev_pop, weights=fitness_proportions, k=k)
        return pop
    except ValueError:
        print(fitness_proportions)
        print('************* ValueError *************')
        print(
            'This is due to choices() requiring a set of weights that sum to greater than 0')
        print('ABORTING RUN, PLEASE TRY AGAIN.')
        exit()


# Assume pop. length of 2
def crossover_selection(pop, indiv_len):
    rand = randint(1, indiv_len - 2)
    crossover = []
    crossover.append(pop[0][:rand] + pop[1][rand:])
    crossover.append(pop[1][:rand] + pop[0][rand:])

    return crossover


def mutation_selection(pop, indiv_len):
    for indiv in pop:
        if random() <= MUTATION_CHANCE:
            # clone = indiv.copy()
            flip_ind = randint(0, indiv_len-1)
            indiv[flip_ind] = 0 if indiv[flip_ind] == 1 else 1
            # print('MUTATION! AT INDEX ' + str(flip_ind), clone, '->', indiv)

    return pop


def create_new_pop(prev_pop, prev_fits, indiv_len):
    # TODO Should I be removing the ELITE individuals once they are selected??
    new_pop = elitism_selection(prev_pop, prev_fits)

    while (len(new_pop) < POP_SIZE):
        # Select TWO parents with roulette wheel
        k = 2 if POP_SIZE - len(new_pop) > 2 else 1
        roulette = roulette_selection(prev_pop, prev_fits, k)

        # apply one-point crossover to them
        crossover = crossover_selection(
            roulette, indiv_len) if len(roulette) == 2 else roulette

        # 5% chance of mutation (flip)
        mutation = mutation_selection(crossover, indiv_len)

        # put the children into the pop
        new_pop += mutation

    return new_pop


def read_file(data_file, names_file):
    data = []

    with open(Path(data_file), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
            #TODO Probs need to strip the last num, it's either 1/2

    # print(len(data[0]))
    sm = 222222
    lg = -222222
    for i, row in enumerate(data):
        v = row[:-1]
        v.sort()

        if v[0] < sm:
            sm = v[0]
            print(i, sm)
        
        if v[-1] > lg:
            lg = v[-1]
            print(i, lg)

    print(sm, lg)

    # print(1.0/30.0)

    return data


if __name__ == '__main__':
    main()
