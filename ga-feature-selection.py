from pathlib import Path
from random import randint, choices, random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
import csv
import sys
from math import log2
from scipy.stats import entropy

NUM_GEN = 100
NUM_FEAT = 30
# POP_SIZE = 100
ELITISM = 0.1
MUTATION_CHANCE = 0.1
ALPHA = 0.1
N_BINS = 30


def main():
    data, classes = read_file('./wbcd.data', './wbcd.names')

    c1 = 0
    c2 = 0
    for i in classes:
        if i == '1':
            c1 += 1
        elif i == '2':
            c2 += 1

    pc1 = c1 / len(classes)
    pc2 = c2 / len(classes)

    print(pc1, pc2, len(classes))

    run_feature_selection(data)

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
    # INITIALISE POPULATION
    pop = [[randint(0, 1) for __ in range(NUM_FEAT)]
           for ___ in range(len(data))]
    pop_fit = [None] * len(data)

    # values = [row[0] for row in data]
    # weights = [row[1] for row in data]

    best_feasible = None
    best_fit = 0

    probs = []
    # for __ in range(NUM_GEN):
    # Calc fitness of current pop
    # for i, ind in enumerate(pop):
    #     total = sum(data[i])
    #     probs.append([x / total for x in data[i]])

    # print(len(probs))
    # print(probs[-1])

    discretizer = KBinsDiscretizer(
        n_bins=N_BINS, encode='ordinal', strategy='kmeans')
    discretizer.fit(data)
    transformed_data = discretizer.transform(data)

    # print('DATA', data[0])
    # print('DISCRETIZED', transformed_data[:][0])

    # fitness = calc_feature_selection(
    #     pop[0], transformed_data[0], 1./float(N_BINS))

    # print(probs[-1])

    # print(prob)

    # pop_fit[i] = calc_feature_selection(ind, prob)
    # Find new best feasible
    # best_ind = max(range(len(pop_fit)), key=pop_fit.__getitem__)
    # if pop_fit[best_ind] > best_fit:
    #     best_feasible = pop[best_ind]
    #     best_fit = pop_fit[best_ind]

    # pop = create_new_pop(pop, pop_fit, individual_len)

    # return best_feasible

##
# Fitness calculation, filter function.
# Probabilities have been districtised.


def calc_feature_selection(subset, dis_individual, prob, ets=1e-15):

    e = -sum([(p) * log2(p) for p in probabilities])

    print(e)

    # print('My entropy', me_e)
    # total = 0
    # print(me_e)
    # for p in probabilities:
    #     print('Prob', p)
    #     curr = p * log2(p)
    #     print('Each', curr)
    #     total += curr
    # total *= -1

    # e = entropy(probabilities, base=2)

    # print('their entropy', e)


    # cond_prob = []
    # selection = []
    # for i in range(len(individual)):
    #     if individual[i]:
    #         cond_prob.append(probabilities[i])
    #         selection.append(i)

    # cond_e = entropy(cond_prob, base=2)

    # val = mutual_info_classif(probabilities, cond_prob)

    # print('their mutual', val)


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
    classes = []

    with open(Path(data_file), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            classes.append(row[-1])

    # print(len(data[0]))
    # sm = 222222
    # lg = -222222
    # for i, row in enumerate(data):
    #     v = row[:-1]
    #     v.sort()

    #     if v[0] < sm:
    #         sm = v[0]
    #         print(i, sm)

    #     if v[-1] > lg:
    #         lg = v[-1]
    #         print(i, lg)

    # print(sm, lg)

    # print(1.0/30.0)

    return classes, data


if __name__ == '__main__':
    main()
