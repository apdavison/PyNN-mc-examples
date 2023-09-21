"""
This script is based on "network_ring.py"
from the "python/example" folder of the Arbor source code, but:
(a) uses plain matplotlib rather than seaborn for plotting
and numpy rather than pandas for saving data;
(b) uses a spike-source cell rather than an event generator
"""

import arbor
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# Construct a cell with the following morphology.
# The soma (at the root of the tree) is marked 's', and
# the end of each branch i is marked 'bi'.
#
#         b1
#        /
# s----b0
#        \
#         b2


def make_cable_cell(gid):
    # (1) Build a segment tree
    tree = arbor.segment_tree()

    # Soma (tag=1) with radius 6 μm, modelled as cylinder of length 2*radius
    s = tree.append(
        arbor.mnpos, arbor.mpoint(-12, 0, 0, 6), arbor.mpoint(0, 0, 0, 6), tag=1
    )

    # (b0) Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
    b0 = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(50, 0, 0, 2), tag=3)

    # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
    # (b1) Radius tapers from 2 to 0.5 μm over the length of the dendrite.
    tree.append(
        b0,
        arbor.mpoint(50, 0, 0, 2),
        arbor.mpoint(50 + 50 / sqrt(2), 50 / sqrt(2), 0, 0.5),
        tag=3,
    )
    # (b2) Constant radius of 1 μm over the length of the dendrite.
    tree.append(
        b0,
        arbor.mpoint(50, 0, 0, 1),
        arbor.mpoint(50 + 50 / sqrt(2), -50 / sqrt(2), 0, 1),
        tag=3,
    )

    # Associate labels to tags
    labels = arbor.label_dict(
        {
            "soma": "(tag 1)",
            "dend": "(tag 3)",
            # (2) Mark location for synapse at the midpoint of branch 1 (the first dendrite).
            "synapse_site": "(location 1 0.5)",
            # Mark the root of the tree.
            "root": "(root)",
        }
    )

    # (3) Create a decor and a cable_cell
    decor = (
        arbor.decor()
        # Put hh dynamics on soma, and passive properties on the dendrites.
        .paint('"soma"', arbor.density("hh")).paint('"dend"', arbor.density("pas"))
        # (4) Attach a single synapse.
        .place('"synapse_site"', arbor.synapse("expsyn"), "syn")
        # Attach a detector with threshold of -10 mV.
        .place('"root"', arbor.threshold_detector(-10), "detector")
    )

    return arbor.cable_cell(tree, decor, labels)


# (5) Create a recipe that generates a network of connected cells.
class ring_recipe(arbor.recipe):
    def __init__(self, ncells):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = ncells

    # (6) The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    def cell_description(self, gid):
        if gid == 0:
            sched = arbor.explicit_schedule([0.5])
            # added to spike source synaptic delay this gives event delivery at 1 ms
            return arbor.spike_source_cell("spike-source", sched)
        else:
            return make_cable_cell(gid)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        if gid == 0:
            return arbor.cell_kind.spike_source
        else:
            return arbor.cell_kind.cable

    # (8) Make a ring network. For each gid, provide a list of incoming connections.
    def connections_on(self, gid):
        connections = []
        if gid > 0:
            src = ((gid - 2) % (self.ncells - 1)) + 1
            w = 0.01  # 0.01 μS on expsyn
            d = 5  # ms delay
            connections.append(arbor.connection((src, "detector"), "syn", w, d))
        if gid == 1:
            w = 0.1   # 0.1 μS on expsyn
            d = 0.5  # ms delay
            connections.append(arbor.connection((0, "spike-source"), "syn", w, d))
        return connections

    # (9) No generators used
    def event_generators(self, gid):
        return []

    # (10) Place a probe at the root of each cell.
    def probes(self, gid):
        if gid > 0:
            return [arbor.cable_probe_membrane_voltage('"root"')]
        else:
            return []

    def global_properties(self, kind):
        if kind == arbor.cell_kind.cable:
            return arbor.neuron_cable_properties()
        # Spike source cells have nothing to report.
        return None


# (11) Instantiate recipe
ncells = 4 + 1   # + 1 for the spike source
recipe = ring_recipe(ncells)

# (12) Create a simulation using the default settings:
# - Use all threads available
# - Use round-robin distribution of cells across groups with one cell per group
# - Use GPU if present
# - No MPI
# Other constructors of simulation can be used to change all of these.
sim = arbor.simulation(recipe)

# (13) Set spike generators to record
sim.record(arbor.spike_recording.all)

# (14) Attach a sampler to the voltage probe on cell 0. Sample rate of 10 sample every ms.
handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(1, ncells)]

# (15) Run simulation for 100 ms
sim.run(100)
print("Simulation finished")

# (16) Print spike times
print("spikes:")
for sp in sim.spikes():
    print(" ", sp)

# (17) Plot the recorded voltages over time.
print("Plotting results ...")
all_data = []
for handle in handles:
    data, meta = sim.samples(handle)[0]
    all_data.append(data)

for i, cell_data in enumerate(all_data):
    plt.plot(cell_data[:, 0], cell_data[:, 1], label=f"Cell {i}")

plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.legend()
plt.savefig("results/ring_network_with_spike_source_arbor_recipe.png")

np.savetxt(
    "results/ring_network_with_spike_source_arbor_recipe.original_data",
    np.vstack([cell_data[:, 1] for cell_data in all_data])
)
