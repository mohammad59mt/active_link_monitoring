import random

class Probes:
    def __init__(self, topo):
        self.topo = topo
        self.switch_switch_set = set() # set of links between switches
        self.host_switch_set = set()   # set of links between a host and a switch
        self.linkBasedPath_arrayOfList = []  # list of selected paths (each path is a list of links)
        self.linkBasedPath_arrayOfSet = []  # list of selected paths (each path is a set of links)
        self.nodeBasedPath_arrayOfList = []  # list of selected paths (each path is an array of nodes)
        self.used_links_set = set()    # set of links that exist in one of the selected paths
        self.paths_that_doesnot_have_a_NotTraversedLink = [] # list of paths that are from a correct source to a correct destination, but when they was found all links
                                                             # included in them was traversed by an already selected path
        self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink = []  # source nodes of the above paths (paths_that_doesnot_have_a_NotTraversedLink)
        self.topo_to_linkset()
    def topo_to_linkset(self):
        ''' switch_switch_set and host_switch_set'''
        for element in self.topo:
            value = self.topo[element]
            # zero as value means that there is not any link between these two nodes
            if value != 0:
                # each "element" is either (MAC, MAC, 's') or (IP, MAC, 'h')
                IP_MAC_Setter = lambda x: (x[0],x[1]) if len(x[0])<len(x[1]) else (x[1],x[0])
                if element[2] == 'h': self.host_switch_set.add(IP_MAC_Setter((element[0], element[1])))
                elif element[2] == 's': self.switch_switch_set.add((element[0], element[1]))
    def reset(self):
        self.linkBasedPath_arrayOfList, self.linkBasedPath_arrayOfSet, self.nodeBasedPath_arrayOfList, self.used_links_set,  = [], [], [], []
    def source_switches(self):
        res = [link[1] for link in self.host_switch_set]
        return res

    @staticmethod
    def loop_exists(path):
        met_nodes = []
        for element in path:
            if element[1] in met_nodes:
                return True
            else: met_nodes.append(element[1])
        return False
    @staticmethod
    def convert_linkBasedPath_to_nodeBasedPath(link_based_path, src):
        path = list(link_based_path)
        node_based_path = [src]
        for _ in range(len(path)):
            for element in path:
                if element[0] == src:
                    src = element[1]
                    node_based_path.append(element[1])
                    path.remove(element)
                    break
        return node_based_path

    def add_host_to_selected_nodeBasedPaths(self):
        for node_based_path in self.nodeBasedPath_arrayOfList:
            link_to_source_host = [link for link in self.host_switch_set if node_based_path[0] in link][0]
            link_to_destination_host = [link for link in self.host_switch_set if node_based_path[-1] in link][0]
            node_based_path.append(link_to_destination_host[0])
            node_based_path.insert(0, link_to_source_host[0])
    def find(self, current_node, remaind_hops, destination_node=None, selected_path=list(),source_node=None):
        if remaind_hops == 0:
            ''' if you have found a valid path add it to the list of selected paths'''
            ''' check if you have reached the destination and this path is not included in the currently selected paths'''
            if current_node == destination_node and len(selected_path) != 0 and not set(selected_path) in self.linkBasedPath_arrayOfSet:
                #if there is a link this path that is not met before add it to the list of selected paths.
                if self.has_a_not_traversed_link(selected_path):
                    self.linkBasedPath_arrayOfList.append(selected_path)
                    self.linkBasedPath_arrayOfSet.append(set(selected_path))
                    self.nodeBasedPath_arrayOfList.append(Probes.convert_linkBasedPath_to_nodeBasedPath(selected_path, source_node))
                    self.used_links_set = set.union(self.used_links_set, selected_path)
                else:
                    self.paths_that_doesnot_have_a_NotTraversedLink.append(selected_path)
                    self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink.append(source_node)
            return
        if destination_node is None: destination_node = current_node
        if source_node is None: source_node = current_node
        for (a,b) in self.switch_switch_set:
            if a == current_node:
                if (a,b) not in selected_path:
                    if not Probes.loop_exists((selected_path+[(a,b)])):
                        self.find(current_node=b, remaind_hops=remaind_hops-1, destination_node=destination_node, selected_path=(selected_path+[(a,b)]), source_node=source_node)
    def has_a_not_traversed_link(self, path_to_check):
        # if the union of used_links_set (used links in the other selected paths) and the path_to_check is bigger than the length of used_links_set this means that
        # there is a not traversed link in the path_to_check
        if len(set.union(self.used_links_set, path_to_check)) > len(self.used_links_set): return True
        else: return False
    def make_test(self, min_val, max_val, floating_pionts):
        random_between_min_max = lambda : (max_val-min_val)*random.random()+min_val
        round_value = lambda x: float(("{0:."+str(floating_pionts)+"f}").format(x))
        link_delays = {lnk:round_value(random_between_min_max()) for lnk in self.switch_switch_set}
        Path_Delay = lambda path: sum([link_delays[(path[i],path[i+1])] for i in range(len(path)-1)])
        path_delays = [Path_Delay(path) for path in self.nodeBasedPath_arrayOfList]
        rounded_path_delays = [round_value(Path_Delay(path)) for path in self.nodeBasedPath_arrayOfList]
        return path_delays, rounded_path_delays, link_delays
    ''' In the above lines, all possible paths will be investigated but only those that have not_traversed_link will be added to result. 
    In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result'''
    def ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(self, max_number_probes_per_link):
        number_of_existing_links = len(self.switch_switch_set)
        number_of_probes = len(self.linkBasedPath_arrayOfList)
        number_of_not_used_paths = len(self.paths_that_doesnot_have_a_NotTraversedLink)
        if number_of_existing_links * max_number_probes_per_link > number_of_probes and number_of_not_used_paths > 0:
            for i in range(min(number_of_not_used_paths, number_of_existing_links-number_of_probes)):
                self.linkBasedPath_arrayOfList.append(self.paths_that_doesnot_have_a_NotTraversedLink[i])
                self.linkBasedPath_arrayOfSet.append(set(self.paths_that_doesnot_have_a_NotTraversedLink[i]))
                self.nodeBasedPath_arrayOfList.append(Probes.convert_linkBasedPath_to_nodeBasedPath(self.paths_that_doesnot_have_a_NotTraversedLink[i], self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink[i]))
                self.used_links_set = set.union(self.used_links_set, self.paths_that_doesnot_have_a_NotTraversedLink[i])
        return

    @staticmethod
    def main(topo, length_of_probes_array, max_number_probes_per_link, debug=True):
        random.seed(400)
        ''' Set the topology'''
        probes = Probes(topo)
        ''' Find possible source switches'''
        source_switch = probes.source_switches()
        ''' Solving the problem'''
        for length_of_probes in length_of_probes_array:
            for src in source_switch:
                ''' find all possible probes starting from src node and have length of length_of_probes'''
                probes.find(src, length_of_probes)
        ''' In the above lines, all possible paths will be investigated but only those that have not_traversed_link will be added to result
            In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result'''
        probes.ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(max_number_probes_per_link)

        if debug:
            path_delays, rounded_path_delays, link_delays = probes.make_test(1, 10, 0)
            print('Links delays: ', link_delays)
            print('Paths delays: ', path_delays)
            print('Rounded paths delays: ', rounded_path_delays)

        probes.add_host_to_selected_nodeBasedPaths()

        if debug:
            print('Link based paths: ', probes.linkBasedPath_arrayOfList)
        print('Node based paths: ', probes.nodeBasedPath_arrayOfList)
        print('Number of probes: ',len(probes.linkBasedPath_arrayOfList))
        print('Number of existing links: ', len(probes.switch_switch_set))
        print('Number of included links: ', len(probes.used_links_set))
        if len(probes.switch_switch_set) > len(probes.used_links_set): print("********ERROR: Some links will not be measured.********\nChange the length_of_probes_array to solve the problem.")
        if len(probes.linkBasedPath_arrayOfList) != len(probes.used_links_set): print("********WARNING: Number of probes is not the same as Number of links.********\nChange the length_of_probes_array to make them equal.\nAt least keep the number of probes more than the number of links")
        return probes.nodeBasedPath_arrayOfList


def heuristic_for_ILP(topo=None, length_of_probes_array=None, max_number_probes_per_link = 2,debug=True):
    if topo is None: print('Topology is not specified; using sample topology ...'); topo = {('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 'h'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 'h'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0b', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 'h'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 'h'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0b', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 'h'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0b', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1}
    if length_of_probes_array is None: print('Array_of_LengthOfProbes is not specified; using [2, 7] ...'); length_of_probes_array = [2,5]
    return Probes.main(topo, length_of_probes_array, max_number_probes_per_link, debug)

# length_of_probes_array = [2,5]
# topo = None
# heuristic_for_ILP(topo=topo, length_of_probes_array=length_of_probes_array, debug=False)
