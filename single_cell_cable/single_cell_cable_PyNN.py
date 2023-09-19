"""

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology
from pyNN.utility import get_simulator

sim, options = get_simulator()

args = {
    "Vm": -65,
    "length": 1000,
    "radius": 1,
    "cm": 0.01,
    "rL": 90,
    "g": 0.001,
    "stimulus_start": 10,
    "stimulus_duration": 0.1,
    "stimulus_amplitude": 1.0,
    "cv_policy_max_extent": 10,
    "dt": 0.001
}

sim.setup(timestep=args["dt"])

cell_class = sim.MultiCompartmentNeuron
cell_class.label = f"{options.simulator.title()}SimpleDendriteExample"
cell_class.ion_channels = {'pas': sim.PassiveLeak}

n_segments = round(args["length"] / args["cv_policy_max_extent"])
#seg_length = args["cv_policy_max_extent"]
diam = 2 * args["radius"]
# cable = [
#     Segment(proximal=P(x=0, y=0, z=0, diameter=diam),
#             distal=P(x=0, y=0, z=seg_length, diameter=diam),
#             name="seg0", id=0)
# ]
# for i in range(1, n_segments):
#     cable.append(
#         Segment(proximal=P(x=0, y=0, z=seg_length * i, diameter=diam),
#                 distal=P(x=0, y=0, z=seg_length * (i + 1), diameter=diam),
#                 name=f"seg{i}", id=i, parent=cable[i - 1])
#     )
cable = Segment(proximal=P(x=0, y=0, z=0, diameter=diam),
                distal=P(x=0, y=0, z=args["length"], diameter=diam),
                name="cable", id=0)

morphology = NeuroMLMorphology(NMLMorphology(segments=[cable]))

cell_type = cell_class(
    morphology=morphology,
    ionic_species={},
    cm=args["cm"] * 100,
    Ra=args["rL"],
    pas={
        "conductance_density": sim.morphology.uniform("all", args["g"]),
        "e_rev": args["Vm"]
    }
)

initial_values = {
    "v": args["Vm"],
}
dendrite = sim.Population(1, cell_type, initial_values=initial_values)

stim = sim.DCSource(
    start=args["stimulus_start"],
    stop=args["stimulus_start"] + args["stimulus_duration"],
    amplitude=args["stimulus_amplitude"]
)
stim.inject_into(dendrite, location=sim.morphology.at_distances("cable", [0]))

# recording_locations = {
#     "seg0": "seg0",
#     "seg10": "seg10",
#     "seg20": "seg20",
#     "seg30": "seg30",
#     "seg40": "seg40",
#     "seg50": "seg50",
#     "seg60": "seg60",
#     "seg70": "seg70",
#     "seg80": "seg80",
#     "seg90": "seg90",
#     "seg99": "seg99",
# }
#recording_locations = sim.morphology.at_distances("cable", np.linspace(0, args["length"], 11))
recording_locations = sim.morphology.at_distances("cable", np.linspace(0, 1, 11))
dendrite.record("v", locations=recording_locations)

sim.run(30)

data = dendrite.get_data().segments[0]

base_filename, ext = os.path.splitext(os.path.basename(__file__))
with np.load("results/single_cell_cable_arbor_recipe.original_data.npz") as fp:
    original_data = fp["arr_0"]
for sig, original_arr in zip(sorted(data.analogsignals, key=lambda s: s.name), original_data):
    plt.plot(original_arr[:, 0], original_arr[:, 1], "b-", lw=3)
    plt.plot(sig.times, sig, "g-", label=sig.name)
    plt.xlim(9.5, 12.5)
    plt.legend()
    plt.title(f"Simulated with pyNN.{options.simulator}")
    plt.savefig(f"results/{base_filename}_{options.simulator}.png")
