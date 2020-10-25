# network_link_monitoring
This library provides functions to generate a set of probes to measure per link delay.
In this project, we propose a solution to measure the Link Delay Vector (LDV) in real-time with a low-overhead approach. In particular, we inject some flows into the network and infer the LDV based on the delay of those flows. To this end, the monitoring flows and their paths should be selected minimizing the network monitoring overhead. In this respect, the challenging issue is to select a proper combination of flows such that by knowing their delay it is possible to solve a set of linear equation and obtain a unique LDV. This combination of monitoring flows should be optimal according to some criteria and should met some feasibility constraints. The library deveolped here, is based on our proposed mathematical formulation to select the optimal combination of flows, in form of Integer Linear Programming (ILP) problem. As a further step, we propose a meta-heuristic algorithm to solve the above-mentioned equations and infer the LDV.

## To choose the set of probes
''' specify the allowed length of probes, e.g., [2, 5] means that probes with length of 2 or 5 are allowed '''
length_of_probes_array = [2,5]
''' network topology '''
topo = {('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 1, ..., ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1}
''' The maximum number of probes can be max_number_probes_per_link * number_of_links'''
max_number_probes_per_link = 2
''' invoke the function ''' 
heuristic_for_ILP(topo=topo, length_of_probes_array=length_of_probes_array, max_number_probes_per_link = max_number_probes_per_link, debug=False)

## Inferring per link delays
After injecting the probes to the network and measuring the end-to-end delays follow these lines:
array_of_delays = [7, 11, 18, ..., 29, 38, 30]
node_based_path_array = [['00:00:00:00:00:02',..., '00:00:00:00:00:02'],...]
link_delay_measurement_PSO(array_of_delays, node_based_path_array, debug=True)
