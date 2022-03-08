import numpy as np
import os
import matplotlib.pyplot as plt

# @Author : Labyrinthine Leo
# @Time   : 2021.06.06
# @fun    : Testing the archit entropy

def archit_entropy_value(archit_pop):
    """
    Computing the Architecture-Entropy value.
    :param archit_pop: architecture population(every element is a encode.)
    :return:
    """
    pop_size = len(archit_pop)
    # Get the encode of every individual architecture
    pop_encodes = archit_pop
    # Get the Normal Cell's density and Reduction Cell's density of each individual
    pop_fitness = [[np.array(ind[0][0]).sum(), np.array(ind[1][0]).sum()] for ind in pop_encodes]
    # Set the Space Granularity
    nc_SpaceGran, rc_SpaceGran = 3, 2
    nc_space_dict = {i:0 for i in range(6, 27, 3)} # compute the frequency
    rc_space_dict = {i:0 for i in range(4, 14, 2)}

    for fit in pop_fitness:
        nc_num = fit[0]
        rc_num = fit[1]

        for spc in nc_space_dict.keys(): #
            if spc <= nc_num <= (spc + nc_SpaceGran - 1):
                nc_space_dict[spc] += 1

        for spc in rc_space_dict.keys(): #
            if spc <= rc_num <= (spc + rc_SpaceGran - 1):
                rc_space_dict[spc] += 1
    nc_AE = 0
    rc_AE = 0
    for i in nc_space_dict.keys():
        Pi = nc_space_dict[i]/pop_size
        if Pi != 0:
            nc_AE += (-Pi * np.log(Pi))
    for i in rc_space_dict.keys():
        Pi = rc_space_dict[i]/pop_size
        if Pi != 0:
            rc_AE += (-Pi * np.log(Pi))

    Archit_Entropy = 0.6*nc_AE + 0.4*rc_AE

    return Archit_Entropy