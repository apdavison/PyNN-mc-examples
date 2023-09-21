#!/usr/bin/env python3

from math import sqrt
import numpy as np
import neo
import quantities as pq
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.parameters import IonicSpecies
from pyNN.morphology import NeuroMLMorphology
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel


sim, options = get_simulator()
m = sim.morphology

sim.setup(timestep=0.025)

class TwoBranchNeuron(sim.MultiCompartmentNeuron):
    ion_channels = {
        'na': sim.NaChannel,
        'kdr': sim.KdrChannel,
        'pas': sim.PassiveLeak
    }
    post_synaptic_entities = {
        'syn': sim.CondExpPostSynapticResponse
    }

soma = Segment(proximal=P(x=-12, y=0, z=0, diameter=12),
               distal=P(x=0, y=0, z=0, diameter=12),
               name="soma", id=0)
dend = [
    Segment(proximal=P(x=0, y=0, z=0, diameter=4),
            distal=P(x=50, y=0, z=0, diameter=4),
            name="b0", id=1, parent=soma),
    Segment(proximal=P(x=50, y=0, z=0, diameter=4),
            distal=P(x=50 + 50 / sqrt(2), y=50 / sqrt(2), z=0, diameter=1),
            name="b1", id=2),
    Segment(proximal=P(x=50, y=0, z=0, diameter=2),
            distal=P(x=50 + 50 / sqrt(2), y=-50 / sqrt(2), z=0, diameter=2),
            name="b2", id=3)
]
dend[1].parent = dend[0]
dend[2].parent = dend[1]

morphology = NeuroMLMorphology(NMLMorphology(segments=[soma] + dend))

cell_type = TwoBranchNeuron(
    morphology=morphology,
    ionic_species={
        "na": IonicSpecies("na", reversal_potential=50.0),
        "k": IonicSpecies("k", reversal_potential=-77.0)
    },
    cm=1.0,
    Ra=35.4,
    na={"conductance_density": m.uniform("soma", 0.120)},
    kdr={"conductance_density": m.uniform("soma", 0.036)},
    pas={
        #"conductance_density": 0.0003,
        #"e_rev": -54.3
        "conductance_density": m.uniform(m.dendrites(), 0.001),
        "e_rev": m.uniform(m.dendrites(), -70)
    },
    syn={
        "e_syn": 0.0,
        "tau_syn": 2.0,  # unsure of this value - check Arbor defaults
        "locations": m.centre(m.with_label("b1"))
    }
)

initial_values = {
    "v": -65,
    "na.m": 0.09364195,  # update these to match v=-65
    "na.h": 0.41815053,  # based on the INITIAL block in hh.mod
    "kdr.n": 0.39626825
}
cells = sim.Population(4, cell_type, initial_values=initial_values)

# event emitted at 0.5 ms is delivered at 1.0 ms due to 0.5 ms transmission delay
event_generator = sim.Population(1, sim.SpikeSourceArray(spike_times=[0.5]))

cells.record("v", locations="soma", sampling_interval=0.1)
cells.record("spikes")

weight = 0.01  # μS
delay = 5  # ms


inputs = sim.Projection(
    event_generator,
    cells,
    sim.FromListConnector(
        [(0, 0, 0.1, 0.5)],
        location_selector="all"
    ),
    sim.StaticSynapse(),
    receptor_type="syn",
)
ring = sim.Projection(
    cells, cells,
    sim.FromListConnector(
        [(i, (i + 1) % cells.size, weight, delay)
         for i in range(cells.size)],
        location_selector="all"
    ),
    sim.StaticSynapse(),
    receptor_type="syn"
)

sim.run(100)

data = cells.get_data().segments[0]
#vm = data.filter(name="soma.v")
spikes = data.spiketrains

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes.multiplexed[1]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

original_data = np.loadtxt("results/ring_network_arbor_recipe.original_data")
original_sig = neo.AnalogSignal(original_data.T, sampling_period=0.1 * pq.ms, units="mV", name="original data")


Figure(
        Panel(data.filter(name="soma.v")[0],
              ylabel="Membrane potential, soma (mV)",
              yticks=True, ylim=(-80, 40)),
        Panel(original_sig,
              ylabel="Membrane potential, soma (mV)",
              yticks=True, ylim=(-80, 40),
              xticks=True, xlabel="Time (ms)"),
        # Panel(data.filter(name='soma.na.m')[0],
        #       ylabel="m, soma",
        #       yticks=True, ylim=(0, 1)),
        #  Panel(data.filter(name='soma.na.h')[0],
        #        xticks=True, xlabel="Time (ms)",
        #        ylabel="h, soma",
        #        yticks=True, ylim=(0, 1)),
        # Panel(data.filter(name='soma.kdr.n')[0],
        #       ylabel="n, soma",
        #       xticks=True, xlabel="Time (ms)",
        #       yticks=True, ylim=(0, 1)),
        title="Ring network",
        annotations=f"Simulated with {options.simulator.upper()}"
    ).save(f"results/ring_network_PyNN_{options.simulator}.png")