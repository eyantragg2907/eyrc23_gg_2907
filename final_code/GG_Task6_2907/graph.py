""" 
* Team Id:              2907
* Author List:          Arnav Rustagi
* Filename:             graph.py
* Theme:                GeoGuide (GG)
* Functions:            None
* Global Variables:     None
"""

from constants import GRAPH_DATA_FILENAME, reverse_pose


class Graph:
    """
    * Class Name:   Graph
    * Use:          Represents the graph structure used for path planning.
    """

    """ 
    * Function Name:    __init__
    * Input:            filename: str (file name to open)
    * Output:           None
    * Logic:            Makes a new Graph object.
    * Example Call:     Graph(filename)
    """
    def __init__(self, filename=GRAPH_DATA_FILENAME):
        self.nodes = {}
        self.goal_nodes = set()
        self.load_from_file(filename)

    """ 
    * Function Name:    load_from_file
    * Input:            filename: str (file name to open)
    * Output:           None
    * Logic:            Read's the graph data from the file and loads it into the graph structure.
    * Example Call:     Graph.load_from_file(filename)
    """
    def load_from_file(self, filename: str) -> None:

        # Read the file line by line
        for line in open(filename).read().splitlines():
            # Split the line into words
            words = line.split()
            N = len(words)

            # If the number of words is not 2, 4 or 5, then continue to the next line
            if N not in (2, 4, 5):
                continue
            if N == 5:  # single way link
                node_1, node_2, direction, distance, pose = words
                self.add_edge(
                    node_1, node_2, eval(direction), float(distance), eval(pose)
                )
            if N == 4:  # initialising an edge
                node_1, node_2, direction, distance = words
                self.add_edge(node_1, node_2, eval(direction), float(distance))
                self.add_edge(
                    node_2, node_1, reverse_pose(eval(direction)), float(distance)
                )
            if N == 2:  # initialising goal node
                self.goal_nodes.add(words[0])

    """ 
    * Function Name:    add_edge
    * Input:            node_1: str (name of the first node), node_2: str (name of the second node), 
                        direction: int (direction to reach node_2 from node_1), distance: float (distance between the nodes),
                        pose: int (pose of the robot at node_2)
    * Output:           None
    * Logic:            Add's an edge in the graph between specified nodes with the 
                            specified direction, distance and pose.
    * Example Call:     Graph.add_edge(filename)
    """
    def add_edge(
        self, node_1: str, node_2: str, direction: int, distance: float, pose=None
    ) -> None:
        
        # If pose is not specified, then set it to direction
        if pose is None:
            pose = direction
        
        # else add the edge in the graph
        if node_1 not in self.nodes:
            self.nodes[node_1] = []
        self.nodes[node_1].append((node_2, direction, distance, pose))

    """ 
    * Function Name:    __len__
    * Input:            None
    * Output:           int: length of the graph
    * Logic:            Return's the length of the graph (number of nodes).
    * Example Call:     len(Graph())
    """
    def __len__(self) -> int:
        return len(self.nodes)

if __name__ == "__main__":

    # code to test the graph class
    graph = Graph()
    print(graph.nodes)
