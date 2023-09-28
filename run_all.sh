cd single_cell
mkdir -p results
python single_cell_arbor_recipe.py
#python single_cell_arbor_PyNN_conversion_step1.py
#python single_cell_arbor_PyNN_conversion_step2.py
#python single_cell_arbor_PyNN_conversion_step3.py
python single_cell_arbor_PyNN_conversion_step4.py
python single_cell_PyNN.py arbor
python single_cell_PyNN.py neuron

cd ../single_cell_cable
mkdir -p results
python single_cell_cable_arbor_recipe.py
python single_cell_cable_PyNN.py arbor
python single_cell_cable_PyNN.py neuron

cd ../two_cells
mkdir -p results
python two_cells_arbor_recipe.py
python two_cells_PyNN.py arbor
python two_cells_PyNN.py neuron

cd ../ring_network
mkdir -p results
python ring_network_arbor_recipe.py
python ring_network_PyNN.py arbor
python ring_network_PyNN.py neuron

cd ..
if [[ "$OSTYPE" == "darwin"* ]]; then
    open single_cell/results/single_cell_PyNN_arbor.png single_cell/results/single_cell_PyNN_neuron.png
    open single_cell_cable/results/single_cell_cable_PyNN_arbor.png single_cell_cable/results/single_cell_cable_PyNN_neuron.png
    open two_cells/results/two_cells_PyNN_arbor.png two_cells/results/two_cells_PyNN_neuron.png
    open ring_network/results/ring_network_PyNN_arbor.png ring_network/results/ring_network_PyNN_neuron.png
else
    xdg-open single_cell/results/single_cell_PyNN_arbor.png single_cell/results/single_cell_PyNN_neuron.png
    xdg-open single_cell_cable/results/single_cell_cable_PyNN_arbor.png single_cell_cable/results/single_cell_cable_PyNN_neuron.png
    xdg-open two_cells/results/two_cells_PyNN_arbor.png two_cells/results/two_cells_PyNN_neuron.png
    xdg-open ring_network/results/ring_network_PyNN_arbor.png ring_network/results/ring_network_PyNN_neuron.png
fi