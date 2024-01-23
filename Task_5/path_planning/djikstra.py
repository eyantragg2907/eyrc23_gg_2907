from constants import *
from graph import Graph


class Path(object):
    def __init__(self, nodes=[], cost=0, instructions = ""):
        self.nodes = nodes
        self.cost = cost
        self.instructions = instructions

    def __add__(self, data):
        (node, cost, instructions) = data
        path = Path(self.nodes[:], self.cost, self.instructions)
        path.nodes.append(node)
        path.cost += cost
        path.instructions += instructions

        return path

    def __len__(self):
        return len(self.nodes)
    
    def __str__(self):
        return f"{self.nodes=}, {self.cost=}, {self.instructions=}"

def get_shortest_path(
    graph,
    node_1,
    node_2,
    path=Path(),
    visited=set(),
    robot_pose=INITIAL_POSE,
):
    if len(path) == 0: path.nodes.append(node_1)
    visited = visited.copy()
    visited.add(node_1)
    children = graph.nodes[node_1]
    paths = []

    for node, direction, distance, end_pose in children:
        if node in visited: continue

        # print("",path)
        cum_cost = distance + get_pose_cost(robot_pose, direction)
        turns = robot_pose - direction
        instructions = LEFT_INSTRUCTION * abs(turns) if turns > 0 else RIGHT_INSTRUCTION * abs(turns)
        updated_path = path + (node, cum_cost, instructions+FORWARD_INSTRUCTION)

        if node == node_2:  return [updated_path]

        paths += get_shortest_path(
            graph,
            node,
            node_2,
            path=updated_path,
            visited=visited,
            robot_pose=end_pose,
        )

    return paths



def get_pose_cost(initial_pose, reqd_pose):
    diff = abs(initial_pose - reqd_pose)
    if reqd_pose > initial_pose:
        return RIGHT_TURN_TIME * diff
    elif reqd_pose < initial_pose:
        return LEFT_TURN_TIME * diff
    else:
        return 0

def run_for_events(events_detected):
    graph = Graph()
    eventslist=[]
    for i in events_detected.items():
        if i[1] != None:
            eventslist.append([PRIORITY[i[1]],'E_'+i[0]])
    eventpriority = [event[1] for event in eventslist]
    seq_of_events = ["A", *eventpriority,"A"] 
    # arnav do your magic here
    for i in range(len(seq_of_events)-1):
        print(f"Shortest path from {seq_of_events[i]} to {seq_of_events[i+1]}" )
        paths = get_shortest_path(graph, seq_of_events[i], seq_of_events[i+1])
        print(min(paths, key=lambda x: x.cost))
if __name__ == "__main__":
    # graph = Graph()
    # paths = get_shortest_path(graph, "A", "G")
    # for x in sorted(paths, key=lambda x: x.cost)[::-1]:
    #     print(x)
    events_detected = {"A":"Combat","B":None,"C":None,"D":"Humanitarian Aid and rehabilitation","E":"Fire"} # our model detected this events
    run_for_events(events_detected)