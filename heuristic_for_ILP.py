import random, numpy as np

'''This class tries to find exactly M equations to solve N unknowns. Best practise to have independent Equations, however, they may be dependent
In other words, if a probe meets a new link that is not traversed through previous probes, the new probe would be added to the list of selected probes.
At the end, if the number of probes is not M, some random probes will be added to the list of selected probes'''
class Probes_noSimilarPath:
    def __init__(self, topo):
        self.topo = topo
        self.switch_switch_set = set()  # set of links between switches
        self.host_switch_set = set()  # set of links between a host and a switch
        self.linkBasedPath_arrayOfList = []  # list of selected paths (each path is a list of links)
        self.linkBasedPath_arrayOfSet = []  # list of selected paths (each path is a set of links)
        self.nodeBasedPath_arrayOfList = []  # list of selected paths (each path is an array of nodes)
        self.used_links_set = set()  # set of links that exist in one of the selected paths
        self.paths_that_doesnot_have_a_NotTraversedLink = []  # list of paths that are from a correct source to a correct destination, but when they was found all links
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
                IP_MAC_Setter = lambda x: (x[0], x[1]) if len(x[0]) < len(x[1]) else (x[1], x[0])
                if element[2] == 'h':
                    self.host_switch_set.add(IP_MAC_Setter((element[0], element[1])))
                elif element[2] == 's':
                    self.switch_switch_set.add((element[0], element[1]))
    def reset(self):
        self.linkBasedPath_arrayOfList, self.linkBasedPath_arrayOfSet, self.nodeBasedPath_arrayOfList, self.used_links_set, = [], [], [], []
    def source_switches(self):
        res = [link[1] for link in self.host_switch_set]
        return res
    @staticmethod
    def loop_exists(path):
        met_nodes = []
        for element in path:
            if element[1] in met_nodes:
                return True
            else:
                met_nodes.append(element[1])
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
            # find set of all links that connects the source switch to a host
            list_of_link_to_source_host = [link for link in self.host_switch_set if node_based_path[0] in link]
            # find set of all links that connects the destination switch to a host
            list_of_link_to_destination_host = [link for link in self.host_switch_set if node_based_path[-1] in link]
            # if source and destination are not the same
            if node_based_path[-1] != node_based_path[0]:
                # choose randomly one of the hosts as source (among the hosts that are connected to the source switch)
                link_to_source_host = list_of_link_to_source_host[
                    random.randint(0, len(list_of_link_to_source_host) - 1)]
                # choose randomly one of the hosts as destination (among the hosts that are connected to the destination switch)
                link_to_destination_host = list_of_link_to_destination_host[
                    random.randint(0, len(list_of_link_to_destination_host) - 1)]
                # add the source host to the path
                node_based_path.append(link_to_destination_host[0])
                # add the destination host to the path
                node_based_path.insert(0, link_to_source_host[0])
            elif node_based_path[-1] == node_based_path[0]:
                # choose randomly one of the hosts for source and another one for destination
                link_to_host = list_of_link_to_source_host[random.randint(0, len(list_of_link_to_source_host) - 1)]
                # add the source host to the path
                node_based_path.append(link_to_host[0])
                # add the destination host to the path
                node_based_path.insert(0, link_to_host[0])

    def find(self, current_node, remaind_hops, destination_node=None, selected_path=list(), source_node=None):
        if remaind_hops == 0:
            ''' if you have found a valid path add it to the list of selected paths'''
            ''' check if you have reached the destination and this path is not included in the currently selected paths'''
            if current_node == destination_node and len(selected_path) != 0 and not set(
                    selected_path) in self.linkBasedPath_arrayOfSet:
                # if there is a link in this path that is not traversed (met) before add it to the list of selected paths.
                if self.has_a_not_traversed_link(selected_path):
                    self.linkBasedPath_arrayOfList.append(selected_path)
                    self.linkBasedPath_arrayOfSet.append(set(selected_path))
                    self.nodeBasedPath_arrayOfList.append(
                        Probes_noSimilarPath.convert_linkBasedPath_to_nodeBasedPath(selected_path, source_node))
                    self.used_links_set = set.union(self.used_links_set, selected_path)
                else:
                    if selected_path not in self.paths_that_doesnot_have_a_NotTraversedLink:
                        self.paths_that_doesnot_have_a_NotTraversedLink.append(selected_path)
                        self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink.append(source_node)
            return
        if destination_node is None: destination_node = current_node
        if source_node is None: source_node = current_node
        for (a, b) in self.switch_switch_set:
            if a == current_node:
                if (a, b) not in selected_path:
                    if not Probes_noSimilarPath.loop_exists((selected_path + [(a, b)])):
                        self.find(current_node=b, remaind_hops=remaind_hops - 1, destination_node=destination_node,
                                  selected_path=(selected_path + [(a, b)]), source_node=source_node)
    def has_a_not_traversed_link(self, path_to_check):
        # if the union of used_links_set (used links in the other selected paths) and the path_to_check is bigger than the length of used_links_set this means that
        # there is a not traversed link in the path_to_check
        if len(set.union(self.used_links_set, path_to_check)) > len(self.used_links_set):
            return True
        else:
            return False

    def make_test(self, min_val, max_val, floating_pionts):
        random_between_min_max = lambda: (max_val - min_val) * random.random() + min_val
        round_value = lambda x: float(("{0:." + str(floating_pionts) + "f}").format(x))
        link_delays = {lnk: round_value(random_between_min_max()) for lnk in self.switch_switch_set}
        Path_Delay = lambda path: sum([link_delays[(path[i], path[i + 1])] for i in range(len(path) - 1)])
        path_delays = [Path_Delay(path) for path in self.nodeBasedPath_arrayOfList]
        rounded_path_delays = [round_value(Path_Delay(path)) for path in self.nodeBasedPath_arrayOfList]
        return path_delays, rounded_path_delays, link_delays

    ''' In the above lines, all possible paths will be investigated but only those that have not_traversed_link will be added to result. 
    In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result'''
    def ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(self,
                                                                                                  max_number_probes_per_link=None,
                                                                                                  max_number_probes=None):
        number_of_existing_links = len(self.switch_switch_set)
        number_of_probes = len(self.linkBasedPath_arrayOfList)
        number_of_not_used_paths = len(self.paths_that_doesnot_have_a_NotTraversedLink)

        if max_number_probes is None and max_number_probes_per_link is not None:
            max_number_probes = number_of_existing_links * max_number_probes_per_link
        else:
            max_number_probes = number_of_not_used_paths

        if max_number_probes > number_of_probes and number_of_not_used_paths > 0:
            for i in range(min(number_of_not_used_paths, max_number_probes - number_of_probes)):
                self.linkBasedPath_arrayOfList.append(self.paths_that_doesnot_have_a_NotTraversedLink[i])
                self.linkBasedPath_arrayOfSet.append(set(self.paths_that_doesnot_have_a_NotTraversedLink[i]))
                self.nodeBasedPath_arrayOfList.append(Probes_noSimilarPath.convert_linkBasedPath_to_nodeBasedPath(
                    self.paths_that_doesnot_have_a_NotTraversedLink[i],
                    self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink[i]))
                self.used_links_set = set.union(self.used_links_set, self.paths_that_doesnot_have_a_NotTraversedLink[i])
        return

    @staticmethod
    def main(topo, length_of_probes_array, max_number_probes_per_link, debug=True, display_outputs=True):
        random.seed(400)
        print(
            '------------------------------------------- Starting to find no-similar probes ------------------------------------------')
        ''' Set the topology'''
        probes = Probes_noSimilarPath(topo)
        ''' Find possible source switches'''
        source_switch = probes.source_switches()
        ''' Solving the problem'''
        for length_of_probes in length_of_probes_array:
            for src in source_switch:
                ''' find all possible probes starting from src node and have length of length_of_probes'''
                probes.find(src, length_of_probes)
        ''' In the above lines, all possible paths will be investigated but only those that have not_traversed_link will be added to result
            In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result'''
        probes.ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(
            max_number_probes_per_link)

        if debug:
            path_delays, rounded_path_delays, link_delays = probes.make_test(1, 10, 0)
            print('Links delays: ', link_delays)
            print('Paths delays: ', path_delays)
            print('Rounded paths delays: ', rounded_path_delays)

        probes.add_host_to_selected_nodeBasedPaths()

        if debug:
            print('Link based paths: ', probes.linkBasedPath_arrayOfList)
        if display_outputs: print('Node based paths (noSimilarPath): ', probes.nodeBasedPath_arrayOfList)
        print('Number of probes (noSimilarPath): ', len(probes.linkBasedPath_arrayOfList))
        print('Number of existing links: ', len(probes.switch_switch_set))
        print('Number of included links using noSimilarPath method: ', len(probes.used_links_set))
        if len(probes.switch_switch_set) > len(probes.used_links_set): print(
            "********ERROR: Some links will not be measured.********\nChange the length_of_probes_array to solve the problem.")
        if len(probes.linkBasedPath_arrayOfList) != len(probes.used_links_set): print(
            "********WARNING: Number of probes is not the same as Number of links.********\nChange the length_of_probes_array to make them equal.\nAt least keep the number of probes more than the number of links")
        print(
            '------------------------------------------- End of find no-similar probes section ------------------------------------------')
        return probes.nodeBasedPath_arrayOfList


'''This class tries to find exactly N linearly-independent equations to solve exactly N unknowns. 
In other words, if we have N links, this class finds exactly N probes where the route selected for these probes are linearly independent.
Therefore, we will have a Square_matrix that is fully_ranked.'''
class Probes_SquareFullRank:
    def __init__(self, topo):
        self.topo = topo
        self.switch_switch_set = set()  # set of links between switches
        self.switch_switch_list = []  # array of links between switches
        self.host_switch_set = set()  # set of links between a host and a switch
        self.linkBasedPath_arrayOfList = []  # list of selected paths (each path is a list of links)
        self.linkBasedPath_arrayOfSet = []  # list of selected paths (each path is a set of links)
        self.nodeBasedPath_arrayOfList = []  # list of selected paths (each path is an array of nodes)
        self.used_links_set = set()  # set of links that exist in one of the selected paths
        self.paths_that_are_not_linearly_independent = []  # list of paths that are from a correct source to a correct destination, but when they were found they
        # weren't linearly-independent of existing paths (paths that were found before this new path)
        self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink = []  # source nodes of the above paths (paths_that_are_not_linearly_independent)
        self.topo_to_linkset()
    def topo_to_linkset(self):
        ''' switch_switch_set and host_switch_set'''
        for element in self.topo:
            value = self.topo[element]
            # zero as value means that there is not any link between these two nodes
            if value != 0:
                # each "element" is either (MAC, MAC, 's') or (IP, MAC, 'h')
                IP_MAC_Setter = lambda x: (x[0], x[1]) if len(x[0]) < len(x[1]) else (x[1], x[0])
                if element[2] == 'h':
                    self.host_switch_set.add(IP_MAC_Setter((element[0], element[1])))
                elif element[2] == 's':
                    self.switch_switch_set.add((element[0], element[1]))
        self.switch_switch_list = list(self.switch_switch_set)
    def reset(self):
        self.linkBasedPath_arrayOfList, self.linkBasedPath_arrayOfSet, self.nodeBasedPath_arrayOfList, self.used_links_set, = [], [], [], []
    def source_switches(self):
        res = [link[1] for link in self.host_switch_set]
        return res
    @staticmethod
    def loop_exists(path):
        met_nodes = []
        for element in path:
            if element[1] in met_nodes:
                return True
            else:
                met_nodes.append(element[1])
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
            # find set of all links that connects the source switch to a host
            list_of_link_to_source_host = [link for link in self.host_switch_set if node_based_path[0] in link]
            # find set of all links that connects the destination switch to a host
            list_of_link_to_destination_host = [link for link in self.host_switch_set if node_based_path[-1] in link]
            # if source and destination are not the same
            if node_based_path[-1] != node_based_path[0]:
                # choose randomly one of the hosts as source (among the hosts that are connected to the source switch)
                link_to_source_host = list_of_link_to_source_host[
                    random.randint(0, len(list_of_link_to_source_host) - 1)]
                # choose randomly one of the hosts as destination (among the hosts that are connected to the destination switch)
                link_to_destination_host = list_of_link_to_destination_host[
                    random.randint(0, len(list_of_link_to_destination_host) - 1)]
                # add the source host to the path
                node_based_path.append(link_to_destination_host[0])
                # add the destination host to the path
                node_based_path.insert(0, link_to_source_host[0])
            elif node_based_path[-1] == node_based_path[0]:
                # choose randomly one of the hosts for source and another one for destination
                link_to_host = list_of_link_to_source_host[random.randint(0, len(list_of_link_to_source_host) - 1)]
                # add the source host to the path
                node_based_path.append(link_to_host[0])
                # add the destination host to the path
                node_based_path.insert(0, link_to_host[0])

    """ Find linearly-independent probes. At the same time, adds all possible probes into paths_that_are_not_linearly_independent"""
    def find_linearly_independent_probes(self, current_node, remaind_hops, destination_node=None, selected_path=list(), source_node=None, max_number_of_probes=None):
        """ Greedy algorithm seeking for probes that are linearly independent of each other"""
        ''' if we have found enough probes just stop searching new probes '''
        if max_number_of_probes is not None and len(max_number_of_probes) <= len(self.linkBasedPath_arrayOfList): return
        ''' if we have found N linearly-independent probes for N link (and there is not limit on the required probes) just stop searching new probes'''
        if len(self.switch_switch_set) == len(self.linkBasedPath_arrayOfList) and max_number_of_probes is None: return
        ''' if the flow length is exactly the requested length'''
        if remaind_hops == 0:
            ''' if you have found a valid path add it to the list of selected paths or put it in a temporary list'''
            # To this end, first check if you have reached the destination and this path is not included in the currently selected paths
            if current_node == destination_node and len(selected_path) != 0 and not set(selected_path) in self.linkBasedPath_arrayOfSet:
                # Now, if this path is linearly independent of previousely selected path then add it to the list of selected paths.
                if self.is_linearly_independent_of_other_selected_pathes(selected_path):
                    self.linkBasedPath_arrayOfList.append(selected_path)
                    self.linkBasedPath_arrayOfSet.append(set(selected_path))
                    self.nodeBasedPath_arrayOfList.append(Probes_SquareFullRank.convert_linkBasedPath_to_nodeBasedPath(selected_path, source_node))
                    self.used_links_set = set.union(self.used_links_set, selected_path)
                # else, if this path has not been found before, put it in paths_that_are_not_linearly_independent
                elif selected_path not in self.paths_that_are_not_linearly_independent:
                    self.paths_that_are_not_linearly_independent.append(selected_path)
                    self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink.append(source_node)
            return
        if destination_node is None: destination_node = current_node
        if source_node is None: source_node = current_node
        for (a, b) in self.switch_switch_set:
            if a == current_node:
                if (a, b) not in selected_path:
                    if not Probes_SquareFullRank.loop_exists((selected_path + [(a, b)])):
                        self.find_linearly_independent_probes(current_node=b, remaind_hops=remaind_hops - 1, destination_node=destination_node,
                                  selected_path=(selected_path + [(a, b)]), source_node=source_node)
    def is_linearly_independent_of_other_selected_pathes(self, linkBasedPath):
        #                                                '00:01'->'00:01'    '00:01'->'00:02'    '00:02'->'00:01'    '00:02'->'00:02'
        # [('00:01', '00:02'), ('00:02', '00:01')] ==>   [       0                     1                   1                   0        ]
        def convert_linkBasedPath_to_rowOfArray(linkBasedPath):
            res = [0 for _ in range(len(self.switch_switch_list))]
            for link in linkBasedPath:
                res[self.switch_switch_list.index(link)] = 1
            return res

        a = [convert_linkBasedPath_to_rowOfArray(path) for path in self.linkBasedPath_arrayOfList]
        a.append(convert_linkBasedPath_to_rowOfArray(linkBasedPath))
        if np.linalg.matrix_rank(a) == len(a):
            return True
        else:
            return False

    ''' In find_probes_traversing_aNotMetLink, all possible paths will be investigated but only those that have not_traversed_link will be added to result. 
    In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result'''
    def ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(self, max_number_probes=None):
        ''' if there is no criteria on the number of probes, return'''
        if max_number_probes is None: return

        number_of_probes = len(self.linkBasedPath_arrayOfList)
        number_of_not_used_paths = len(self.paths_that_are_not_linearly_independent)
        ''' if the request number of probes is more than found probes, add some new probes (if there is any)'''
        if max_number_probes > number_of_probes and number_of_not_used_paths > 0:
            for i in range(min(number_of_not_used_paths, max_number_probes - number_of_probes)):
                self.linkBasedPath_arrayOfList.append(self.paths_that_are_not_linearly_independent[i])
                self.linkBasedPath_arrayOfSet.append(set(self.paths_that_are_not_linearly_independent[i]))
                self.nodeBasedPath_arrayOfList.append(Probes_noSimilarPath.convert_linkBasedPath_to_nodeBasedPath(self.paths_that_are_not_linearly_independent[i], self.sourceNodeOf_pathsThatDoesnothaveA_NotTraversedLink[i]))
                self.used_links_set = set.union(self.used_links_set, self.paths_that_are_not_linearly_independent[i])
        return
    """ max_number_of_probes=None and ratio_of_numberOfProbes_to_numberOfLinks=None means that only linearly_independent routes should be selected. 
    If max_number_of_probes is a number, then exactly max_number_of_probes probes would be returned unless it is lower than 
    the number of linearly-independent probes. 
    If ratio_of_numberOfProbes_to_numberOfLinks is a number then ratio_of_numberOfProbes_to_numberOfLinks*number_of_links probes would be returned.
    If both ratio_of_numberOfProbes_to_numberOfLinks and max_number_of_probes have value, ratio_of_numberOfProbes_to_numberOfLinks will be ignored.
    When the number of probes is pre-specified, first linearly_independent probes will be selected. If they 
    are not enough, the remaining probes will be random unique probes"""
    @staticmethod
    def main(topo, length_of_probes_array, display_outputs=True, ratio_of_numberOfProbes_to_numberOfLinks=None, max_number_probes=None):
        random.seed(400)
        ''' Set the topology'''
        probes = Probes_SquareFullRank(topo)
        ''' Find possible source switches'''
        source_switch = probes.source_switches()
        ''' Pre-processing'''
        number_of_existing_links = len(probes.switch_switch_set)
        if max_number_probes is None and ratio_of_numberOfProbes_to_numberOfLinks is not None:
            max_number_probes = number_of_existing_links * ratio_of_numberOfProbes_to_numberOfLinks
        # -------------------------------------- Starting to find Linearly independent probes -------------------------------------
        ''' Solving the problem: Finding linearly-independent probes'''
        for length_of_probes in length_of_probes_array:
            for src in source_switch:
                ''' find all possible probes starting from src node and have length of length_of_probes'''
                # To this end, all possible paths will be investigated but only those that are linearly independent will be added to result
                probes.find_linearly_independent_probes(src, length_of_probes)
        # -------------------------------------- End of Linearly independent probes section -----------------------------------------
        if display_outputs: print('-------------------------------------- Starting to find Linearly independent probes -------------------------------------')
        if display_outputs: print('Linearly-Independent Link based paths: ', probes.linkBasedPath_arrayOfList)
        if display_outputs: print('Node based Linearly-Independent paths: ', probes.nodeBasedPath_arrayOfList)
        print('Number of Linearly-Independent probes: ', len(probes.linkBasedPath_arrayOfList))
        if display_outputs: print('Number of existing links: ', len(probes.switch_switch_set))
        print('Number of included links in Linearly-Independent routes: ', len(probes.used_links_set))
        if display_outputs: print('-------------------------------------- End of Linearly independent probes section -----------------------------------------')

        ''' Check if the number of probes is less than requested number of probes, add some random probes'''
        # In the above lines, all possible paths will be investigated but only linearly-independent ones will be added to result
        # In the following function, if the number of selected paths is not enough, those paths (that doesn't have not_travesed_link) will be added to the result
        probes.ifNumberOfSelectedPathsIsNotEnough_addNewPathsFrom_pathsThatDoesnotHaveA_NotTraversedLink(max_number_probes)

        ''' Add source and destination hosts. Till here, only the switches were specified'''
        probes.add_host_to_selected_nodeBasedPaths()

        ''' Print the outputs'''
        if display_outputs: print('Linearly-Independent Link based paths: ', probes.linkBasedPath_arrayOfList)
        if display_outputs: print('Node based Linearly-Independent paths: ', probes.nodeBasedPath_arrayOfList)
        print('Number of all probes: ', len(probes.linkBasedPath_arrayOfList))
        print('Number of existing links: ', len(probes.switch_switch_set))
        print('Number of included links: ', len(probes.used_links_set))
        if display_outputs and len(probes.switch_switch_set) > len(probes.used_links_set): print(
            "********ERROR: Some links will not be measured.********\nChange the length_of_probes_array or 'number of probes' to solve the problem.")
        if display_outputs and len(probes.linkBasedPath_arrayOfList) != len(probes.used_links_set): print(
            "WARNING: Number of Linearly-independent probes is not the same as Number of links (Changing the length_of_probes_array or 'number of probes' may help).")

        return probes.nodeBasedPath_arrayOfList


# specify the allowed length of probes, e.g., [2, 5] means that probes with length of 2 or 5 are allowed
length_of_probes_array = [2, 5]
# network topology
topo = {('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 1,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:06', 's'): 1,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 'h'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:02', 's'): 1,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:01', 's'): 1,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:08', 's'): 1,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:08', 's'): 1,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:07', 's'): 1,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0a', 's'): 1,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 'h'): 1,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', 's'): 1,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:09', 's'): 1,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 's'): 1,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:06', 's'): 1,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0b', 's'): 1,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', 's'): 1,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', 's'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 'h'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:01', 's'): 1,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:06', 's'): 1,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0a', 's'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', 's'): 0,
        ('00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 'h'): 1,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0a', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:06', 's'): 1,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:05', 's'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0b', 's'): 1,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 's'): 1,
        ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 'h'): 1,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:04', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:05', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:08', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0b', 's'): 1,
        ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04', 's'): 1,
        ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:03', 's'): 1,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:03', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:08', 's'): 1,
        ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:06', 's'): 0,
        ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b', 's'): 0,
        ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', 's'): 1,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', 's'): 1,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:09', 's'): 0,
        ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:01', 's'): 0,
        ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:07', 's'): 0,
        ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1}
# REIRIS Topology
# topo = {('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0c', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:10', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:11', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:12', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:08', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:10', 's'): 1, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0b', 's'): 1, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0c', '00:00:00:00:00:00:00:0e', 's'): 1, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0c', 's'): 1, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:11', 's'): 1, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0f', 's'): 1, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:10', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:10', 's'): 1, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0d', 's'): 1, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:11', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:03', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0c', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0b', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:03', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:05', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:04', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:05', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:11', 's'): 1, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0d', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:01', 's'): 1, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:02', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:02', 's'): 1, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:01', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:10', 's'): 1, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0f', '00:00:00:00:00:00:00:0e', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:06', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:07', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0b', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0d', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:09', 's'): 1, ('00:00:00:00:00:00:00:0a', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:04', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:06', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:12', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:08', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0c', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0f', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:07', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0a', 's'): 1, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:09', '00:00:00:00:00:00:00:0e', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:12', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:08', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0c', 's'): 1, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:10', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:11', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:03', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0b', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:04', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:05', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0d', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:02', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:01', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0f', 's'): 1, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:07', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0a', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:06', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:09', 's'): 0, ('00:00:00:00:00:00:00:0e', '00:00:00:00:00:00:00:0e', 's'): 0, ('10.0.0.8', '00:00:00:00:00:00:00:08', 'h'): 1, ('10.0.0.2', '00:00:00:00:00:00:00:02', 'h'): 1, ('10.0.0.1', '00:00:00:00:00:00:00:01', 'h'): 1, ('10.0.0.3', '00:00:00:00:00:00:00:03', 'h'): 1, ('10.0.0.4', '00:00:00:00:00:00:00:04', 'h'): 1, ('10.0.0.5', '00:00:00:00:00:00:00:05', 'h'): 1, ('10.0.0.6', '00:00:00:00:00:00:00:06', 'h'): 1, ('10.0.0.7', '00:00:00:00:00:00:00:07', 'h'): 1, ('10.0.0.9', '00:00:00:00:00:00:00:09', 'h'): 1, ('10.0.0.10', '00:00:00:00:00:00:00:0a', 'h'): 1, ('10.0.0.11', '00:00:00:00:00:00:00:0b', 'h'): 1, ('10.0.0.12', '00:00:00:00:00:00:00:0c', 'h'): 1,('10.0.0.13', '00:00:00:00:00:00:00:0d', 'h'): 1,('10.0.0.14', '00:00:00:00:00:00:00:0e', 'h'): 1,('10.0.0.15', '00:00:00:00:00:00:00:0f', 'h'): 1,('10.0.0.16', '00:00:00:00:00:00:00:10', 'h'): 1,('10.0.0.17', '00:00:00:00:00:00:00:11', 'h'): 1,('10.0.0.18', '00:00:00:00:00:00:00:12', 'h'): 1}
ratio_of_numberOfProbes_to_numberOfLinks = None
max_number_probes = None

nodeBasedPath_arrayOfList = Probes_SquareFullRank.main(topo=topo, length_of_probes_array=length_of_probes_array,
                                               ratio_of_numberOfProbes_to_numberOfLinks=ratio_of_numberOfProbes_to_numberOfLinks, max_number_probes=max_number_probes,
                                               display_outputs=False)
