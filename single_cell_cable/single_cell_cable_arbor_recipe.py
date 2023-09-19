"""
This script is essentially identical to "single_cell_cable.py"
from the "python/example" folder of the Arbor source code, but
using plain matplotlib rather than seaborn for plotting
and numpy rather than pandas for saving data.
"""

import os
import arbor
import numpy as np
import matplotlib.pyplot as plt


class Cable(arbor.recipe):
    def __init__(
        self,
        probes,
        Vm,
        length,
        radius,
        cm,
        rL,
        g,
        stimulus_start,
        stimulus_duration,
        stimulus_amplitude,
        cv_policy_max_extent,
    ):
        """
        probes -- list of probes

        Vm -- membrane leak potential
        length -- length of cable in μm
        radius -- radius of cable in μm
        cm -- membrane capacitance in F/m^2
        rL -- axial resistivity in Ω·cm
        g -- membrane conductivity in S/cm^2

        stimulus_start -- start of stimulus in ms
        stimulus_duration -- duration of stimulus in ms
        stimulus_amplitude -- amplitude of stimulus in nA

        cv_policy_max_extent -- maximum extent of control volume in μm
        """

        arbor.recipe.__init__(self)

        self.the_probes = probes

        self.Vm = Vm
        self.length = length
        self.radius = radius
        self.cm = cm
        self.rL = rL
        self.g = g

        self.stimulus_start = stimulus_start
        self.stimulus_duration = stimulus_duration
        self.stimulus_amplitude = stimulus_amplitude

        self.cv_policy_max_extent = cv_policy_max_extent

        self.the_props = arbor.neuron_cable_properties()

    def num_cells(self):
        return 1

    def num_sources(self, gid):
        return 0

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

    def cell_description(self, gid):
        """A high level description of the cell with global identifier gid.

        For example the morphology, synapses and ion channels required
        to build a multi-compartment neuron.
        """

        tree = arbor.segment_tree()

        tree.append(
            arbor.mnpos,
            arbor.mpoint(0, 0, 0, self.radius),
            arbor.mpoint(self.length, 0, 0, self.radius),
            tag=1,
        )

        labels = arbor.label_dict({"cable": "(tag 1)", "start": "(location 0 0)"})

        decor = (
            arbor.decor()
            .set_property(Vm=self.Vm, cm=self.cm, rL=self.rL)
            .paint('"cable"', arbor.density(f"pas/e={self.Vm}", {"g": self.g}))
            .place(
                '"start"',
                arbor.iclamp(
                    self.stimulus_start, self.stimulus_duration, self.stimulus_amplitude
                ),
                "iclamp",
            )
        )

        policy = arbor.cv_policy_max_extent(self.cv_policy_max_extent)
        decor.discretization(policy)

        return arbor.cable_cell(tree, decor, labels)


if __name__ == "__main__":
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
        "cv_policy_max_extent": 10
    }

    # set up membrane voltage probes equidistantly along the dendrites
    probes = [
        arbor.cable_probe_membrane_voltage(f"(location 0 {r})")
        for r in np.linspace(0, 1, 11)
    ]
    recipe = Cable(probes, **args)

    # configure the simulation and handles for the probes
    sim = arbor.simulation(recipe)
    dt = 0.001
    handles = [
        sim.sample((0, i), arbor.regular_schedule(dt)) for i in range(len(probes))
    ]

    # run the simulation for 30 ms
    sim.run(tfinal=30, dt=dt)

    # retrieve the sampled membrane voltages and convert to a pandas DataFrame
    print("Plotting results ...")

    base_filename, ext = os.path.splitext(os.path.basename(__file__))
    samples_list = []
    for probe in range(len(handles)):
        samples, meta = sim.samples(handles[probe])[0]
        samples_list.append(samples)
    for samples in samples_list:
        plt.plot(samples[:, 0], samples[:, 1])
        plt.xlim(9.5, 12.5)
    plt.savefig(f"results/{base_filename}.png")

    np.savez_compressed(f"results/{base_filename}.original_data", samples_list)
