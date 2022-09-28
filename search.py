# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem):

    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    @param SearchProblem
    @return a list of Directions objects
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


#custom helper function, do NOT delete
def getActionsFromNodes(last_node : dict) -> list:

    store_actions = []
    store_actions.append(last_node["ACTION"])
    parent_node = last_node["PARENT"]

    while parent_node != None:

        direction_action = parent_node["ACTION"]
        if direction_action is None:
            break
        #TODO delete print statement
        #print(direction_action)
        store_actions.append(direction_action)
        parent_node = parent_node["PARENT"]

    store_actions.reverse()
    #TODO Delete
    #print(store_actions)
    print("DFS Complete")
    return store_actions





def depthFirstSearch(problem: SearchProblem):
    """
    #NEEDS TO RETURN A LIST OF DIRECTIONS TO GO

    Search the deepest nodes in the search tree first.
    # TODO how to access the tree?

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState()) -> Returns a tuple (int,int) -> (x,y) location of agent?
    print("Is the start a goal?", problem.isGoalState(problem.getStartState())) -> Returns Boolean true or false
    print("Start's successors:", problem.getSuccessors(problem.getStartState())) -> Successors are list of child nodes
    [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    First element is a tuple of int so a state?
    Second element is direction which is action?
    Third is maybe some cost??
    """
    "*** YOUR CODE HERE ***"

    #Copied and modified from figure 3.11 AIMA pg. 82

    ##input: initial_state, successor(), goal_test()
    stored_node_tuple_tracker = []

    ###Initialize the initial-state node: initial node = (initial_state, [])
    #node <- a node with STATE = problem.INITIAL-STATE, PATH-COST = 0

    # TODO delete print statements
    #print("STARTING DFS")

    start_node = {}
    start_node["STATE"] = problem.getStartState()
    start_node["COST"] = 0
    start_node["PARENT"] = None
    start_node["ACTION"] = None

    # ##Initialize the frontier
    frontier = util.Stack()

    # Frontier = A LIFO stack with node as the only element
    frontier.push(start_node)


    # Initialize the explored set
    explored = set()


    # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
    if problem.isGoalState(start_node["STATE"]):
        # TODO delete print statements
        #print("DFS REACHED GOAL STATE")
        # no more directions to return if we achieved goal state
        return []

    # loop do
    while True:
        ##if frontier is empty then return Fail
        if frontier.isEmpty():
            raise RuntimeError("No more nodes to explore")

    # curr_node = last node in the frontier, remove curr_node from the frontier
        current_node = frontier.pop()

        #Add curr_node.state to explored
        # set cannot contain dicts as elements
        explored.add(current_node["STATE"])

        if problem.isGoalState(current_node["STATE"]):
            # TODO delete print statements
            # print("GOAL STATE REACHED")
            return getActionsFromNodes(current_node)
    #
    # 	for each action in problem.ACTIONS(node.STATE) do
        # so get all potential actions i.e. the next nodes
        all_successors = problem.getSuccessors(current_node["STATE"])

        for successor_node in all_successors:
            # child = CHILD-NODE(problem, node, action), also keep track of parent to return a set of directions
            child = {"STATE": successor_node[0], "ACTION": successor_node[1], "COST": successor_node[2],
                     "PARENT": current_node}
            #print(child["COST"])

            #if child.STATE is not in explored or frontier then
            # testing they want us to keep track of node only in explored not frontier
            if child not in frontier.list and child["STATE"] not in explored:
            #if child["STATE"] not in explored:
                # TODO delete print statements
                #print("NODE ADDED")
            #if problem.GOAL-TEST(child.STATE) then return Solution(child)
                frontier.push(child)
                # keep track of path here
                #print(child,current_node)



    #util.raiseNotDefined()

#custom function do NOT delete
def check_queue(queue: list, key: str, expected_val):
    flag = False
    for each in queue:
        if each[key] == expected_val:
            flag = True
            break
    return flag

#custom function do NOT delete
def display_curr_frontier_list(list_in : list, key: str):
    new = []
    for each_dict in list_in:
        new.append(each_dict[key])
    return new

#TODO Move code into function cannot have custom function methods created
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    start_node = {"STATE": problem.getStartState(), "PARENT": None,
                  "ACTION": None, "COST": 0}
    frontier = util.Queue()
    explored = []

    #TODO delete print statements
    print("Current frontier list: " + display_curr_frontier_list(frontier.list, "STATE").__str__())
    print("Explored: ", explored)

    frontier.push(start_node)

    while not frontier.isEmpty():
        current_node = frontier.pop()
        explored.append(current_node["STATE"])

        if problem.isGoalState(current_node["STATE"]):
            break

        for successor_node in problem.getSuccessors(current_node["STATE"]):
            if successor_node[0] not in explored and not check_queue(frontier.list, "STATE",successor_node[0]):
                print("Current frontier list: " + display_curr_frontier_list(frontier.list,"STATE").__str__())
                print("Explored: ", explored)
                print(successor_node[0])
                child_node = {"STATE": successor_node[0], "ACTION": successor_node[1],"COST": successor_node[2],
                     "PARENT": current_node}
                frontier.push(child_node)
    print("BFS Complete")
    #input("Hit enter to continue")
    return getActionsFromNodes(current_node)


def check_heapq(prioQHeap : list, target_node, priority):
    in_heap = False
    better_prio = False
    print("in checkheap")
    print(prioQHeap)
    for each in prioQHeap:
        if each[2] == target_node:
            in_heap = True
            if priority <= each[0]:
                better_prio = True
    return in_heap == True and better_prio == True

def isAlreadyInHeap(heapIn: list, target):
    in_heap = False
    for each in heapIn:
        if each[2] == target:
            in_heap = True
            break
    return in_heap

def hasBetterPrio(heapIn: list, target, priority):
    better_prio = False
    for each in heapIn:
        if each[2] == target and priority <= each[0]:
            better_prio = True
            break
    return better_prio


def getUCSPath(dict_paths: dict, last_node_state, first_node_state):
    path = []

    while last_node_state != first_node_state:
        path.append(dict_paths[last_node_state][1])
        last_node_state = dict_paths[last_node_state][0]

    path.reverse()
    return path



def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #cost of the node will determing priority

    #prioQ only cares about priority and state

    #but we need to track parents too....
    child_to_parent_dict = {}
    # key is child node and parent is value
    child_to_parent_dict[problem.getStartState()] = (None, None, 0)
    first_state = problem.getStartState()
    #initialize start node with state, cost


    #initialize prio queue
    frontier = util.PriorityQueue()

    #initizalize explored
    explored = []

    # cost to self is 0
    frontier.push(problem.getStartState(),0)

    while not frontier.isEmpty():
        current_node = frontier.pop()
        # add popped state to explored
        explored.append(current_node[0])

        # is current state the goal state?
        if problem.isGoalState(current_node[0]):
            break

        try:
            # look at child nodes that contain states from current state
            for successor_node in problem.getSuccessors(current_node[0]):
                # if state is not in explored
                if successor_node[0] not in explored:
                    heap_list = list(frontier.heap)

                    # if the state is not in explored but is in the heap
                    if isAlreadyInHeap(heap_list, successor_node[0]):
                        print("state: " + successor_node[0] + " ALREADY in heaplist: " + heap_list.__str__())
                        if hasBetterPrio(heap_list, successor_node[0], successor_node[2]):
                            # if the current state priority is better than the state's priority in the heap
                            print("UPDATING " + successor_node[0])
                            frontier.update(successor_node[0], successor_node[2])
                    else:
                        # if the state is not in the heap
                        print("state: " + successor_node[0] + " not in heaplist: " + heap_list.__str__())
                        print("pushing " + successor_node[0])
                        frontier.push(successor_node[0], successor_node[2])
                    child_to_parent_dict[successor_node[0]] = current_node[0], successor_node[1], successor_node[2]
        except KeyError:
            continue
    print("UCS Complete")
    return getUCSPath(child_to_parent_dict,current_node[0], first_state)





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
