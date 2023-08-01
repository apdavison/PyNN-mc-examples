"""
Converting "single_cell_arbor_recipe.py" to use PyNN:

(1) construct the tree from a PyNN Morphology object
(2) construct the tree and decor within pyNN/arbor/cells.py
(3) construct a PyNN population from which the recipe retrieves information
(4) simulation and recording is also handled by PyNN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.parameters import IonicSpecies
import pyNN.arbor as sim
from pyNN.morphology import NeuroMLMorphology, uniform


sim.setup(timestep=0.025)

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ArborSingleCellExample"
cell_class.ion_channels = {"na": sim.NaChannel, "kdr": sim.KdrChannel, "pas": sim.PassiveLeak}

soma = Segment(proximal=P(x=-3, y=0, z=0, diameter=6),
               distal=P(x=3, y=0, z=0, diameter=6),
               name="soma", id=0)

morphology = NeuroMLMorphology(NMLMorphology(segments=(soma,)))

cell_type = cell_class(
    morphology=morphology,
    ionic_species={
        "na": IonicSpecies("na", reversal_potential=50.0),
        "k": IonicSpecies("k", reversal_potential=-77.0),
    },
    cm=1.0,
    Ra=35.4,
    na={"conductance_density": 0.120},
    kdr={"conductance_density": 0.036},
    pas={
        "conductance_density": uniform('soma', 0.0003),
        "e_rev": -54.3
    }
)

cells = sim.Population(1, cell_type, initial_values={"v": -40})

stim = sim.DCSource(start=10, stop=12, amplitude=0.8)
stim.inject_into(cells, location="soma")

cells.record("v", locations={"soma": "soma"}, sampling_interval=0.1)
cells.record("spikes")

sim.run(30)

data = cells.get_data().segments[0]
vm = data.analogsignals[0]
spikes = data.spiketrains

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes.multiplexed[1]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

base_filename, ext = os.path.splitext(os.path.basename(__file__))
original_data = np.loadtxt("results/single_cell_arbor_recipe.original_data")
plt.plot(original_data[:, 0], original_data[:, 1], "b-", lw=3, label="original")
plt.plot(vm.times, vm, 'g-', label="PyNN conversion (step 4)")
plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.legend()
plt.savefig(f"results/{base_filename}.png")
