import copy
from pathlib import Path
from random import randint, choices, random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
import csv
from math import log2
from scipy.stats import entropy
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from sklearn.naive_bayes import CategoricalNB


NUM_GEN = 10
POP_SIZE = 50
ELITISM = 0.2
MUTATION_CHANCE = 0.5
N_BINS = 10


def main():
    subsets = []
    
    data, classes, feature_count = read_file('./wbcd.data', './wbcd.names')

    discretizer = KBinsDiscretizer(
        n_bins=N_BINS, encode='ordinal', strategy='kmeans')
    discretizer.fit(data)
    transformed_data = discretizer.transform(data)

    for i in range(1, 5):
        start_filter = datetime.now()
        best_feasible, best_fit, history, best_ent = run_feature_selection(transformed_data, deepcopy(classes), copy(feature_count))
        end_filter = datetime.now()
        subsets.append(best_feasible)
        print('Run', i, 'Time elapsed: ' + str(end_filter - start_filter))

    # for subset in subsets:
    subset = subsets[0]
    num_cols = sum(subset)
    subset_data = [[0 for x in range(num_cols)] for y in range(len(transformed_data))]

    subset_col = 0
    for col, val in enumerate(subset):
        if val == 1:
            for row in range(len(transformed_data)):
                subset_data[row][subset_col] = transformed_data[row][col]
            subset_col += 1

    test_size = 199 #int(len(subset_data)/2)

    clf = CategoricalNB()
    clf.fit(subset_data[:-test_size], classes[:-test_size])
    len(subset_data[-test_size:])
    knn = clf.predict(subset_data[-test_size:])
    print(len(knn))

    right = 0
    test = classes[-test_size:]
    for i, c in enumerate(knn):
        if c == test[i]:
            right += 1
    
    print(str((right / test_size * 100)) + '% Correctly Classified')
    print(best_feasible, best_fit, best_ent)

    plt.plot(list(range(NUM_GEN)), history, label='Mean Fitness')
    plt.legend()
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()



def run_feature_selection(data, classes, feature_count):
    class_probs = calc_class_probs(classes)
    # INITIALISE POPULATION
    pop = [[randint(0, 1) for __ in range(feature_count)]
           for ___ in range(POP_SIZE)]

    pop_fit = [None] * POP_SIZE
    pop_ent = [None] * POP_SIZE

    history = []
    best_feasible = None
    best_fit = float('-inf')


    ent = -sum([prob * log2(prob) for prob in class_probs])

    for __ in range(NUM_GEN):
        print('GEN', __)
        # print(pop[:5])
        for i in range(len(pop)):
            pop_fit[i], pop_ent[i] = calc_mutual_information(pop[i], data, ent, classes)

        # Find new best feasible
        best_ind = max(range(len(pop_fit)), key=pop_fit.__getitem__)
        best_ent = min(pop_ent)

        if pop_ent[best_ind] != best_ent:
            print(pop_ent[best_ind], best_ent)
        history.append(pop_ent[best_ind])
        # print(best_ind, pop_fit[best_ind])
        if pop_fit[best_ind] > best_fit:
            best_feasible = pop[best_ind]
            best_fit = pop_fit[best_ind]

        if pop_ent[best_ind] < best_ent:
            best_ent = pop_ent[best_ind]

        pop = create_new_pop(pop, pop_fit, feature_count)
        pop_fit = [None] * POP_SIZE

    return best_feasible, best_fit, history, best_ent

# Fitness calculation, filter function.
def calc_mutual_information(individual, data, ent, classes):
    # Its not about the rows at all. The individuals are just what features to select. There are 30 features
    rows_count = len(data)
    
    # Count class occurences
    c1 = 0
    c2 = 0
    for i in classes:
        if i == 1:
            c1 += 1
        elif i == 2:
            c2 += 1

    total = 0
    for col, indiv_val in enumerate(individual):
        bins_count = [0] * N_BINS
        class_1_count = [0] * N_BINS
        class_2_count = [0] * N_BINS
        if indiv_val == 1:
            for row in range(len(data)):
                ind = int(data[row][col])
                bins_count[ind] += 1
                if classes[row] == 1:
                    class_1_count[ind] += 1
                elif classes[row] == 2:
                    class_2_count[ind] += 1
            
            for i in range(N_BINS):
                pX = float(bins_count[i]) / float(rows_count)
                pY_1 = float(class_1_count[i]) / float(c1)
                pY_2 = float(class_2_count[i]) / float(c2)

                pY_1_calc = (pX * pY_1 * log2(pY_1)) if pY_1 != 0 else 0
                pY_2_calc = (pX * pY_2 * log2(pY_2)) if pY_2 != 0 else 0
                total += pY_1_calc + pY_2_calc
            
    total *= -1

    # TODO Something isn't right with the total...
    # print(ent - total)

    # YOU WANT LOW ENTROPY!!!!!
    # BUT YOU WANT HIGH MUTUAL INFORMATION (which is the equation below!)
    return ent - total, total


def calc_class_probs(classes):
    c1 = 0
    c2 = 0
    for i in classes:
        if i == 1:
            c1 += 1
        elif i == 2:
            c2 += 1

    pc1 = c1 / len(classes)
    pc2 = c2 / len(classes)

    return [pc1, pc2]


# Deep copy arrays and then do elitism.
def elitism_selection(pop, fits):
    clone = deepcopy(pop)
    clone_fit = deepcopy(fits)
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
            flip_ind = randint(0, indiv_len-1)
            # print(flip_ind, indiv_len, indiv)
            indiv[flip_ind] = 0 if indiv[flip_ind] == 1 else 1

    return pop


def create_new_pop(prev_pop, prev_fits, indiv_len):
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
            classes.append(int(row[-1]))

    with open(Path(names_file), 'r') as f:
        reader = csv.reader(f)
        feature_count = sum(1 for row in reader) - 1 # Remove header row by minusing 1.. lol

    return data, classes, feature_count


if __name__ == '__main__':
    main()
