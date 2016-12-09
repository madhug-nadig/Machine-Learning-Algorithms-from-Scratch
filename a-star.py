#!/usr/bin/env python3

#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									A STAR
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import sys
import queue

# each Node in the Graph
class Node:
    def __init__(self, key, i, j):
        # Utility to easily check for presence in openList/ClosedList
        self.__key = key

        # Priority by which it is inserted to OpenList
        self.__priority = 0.0

        # Distance Travelled till current node
        self.__distFromStart = 0

        # set of neighbor nodes
        self.__neighbors = set()

        # node from which this was visited
        self.__parent = None

        self.__traversable = False

        # required for calculating heuristic
        # also used as key in graph.nodes
        self.__pos = (i, j)

    def addNeighbor(self, node):
        self.__neighbors.add(node)

    def setTraversability(self, isTraversable):
        self.__traversable = isTraversable

    def isTraversable(self):
        return self.__traversable

    def getKey(self):
        return self.__key

    def getPos(self):
        return self.__pos

    def setPriority(self, newPriority):
        self.__priority = newPriority

    def getPriority(self):
        return self.__priority

    def setParent(self, parent):
        self.__parent = parent

    def getParent(self):
        return self.__parent;

    def getDistFromStart(self):
        return self.__distFromStart

    def setDistFromStart(self, distFromStart):
        self.__distFromStart = distFromStart

    def getNeighbors(self):
        return self.__neighbors

    def setKey(self, key):
        self.__key = key

    # Requred by queue.PriorityQueue's use of heapq for comparing two nodes
    def __lt__(self, node):
        return self.__priority < node.getPriority()

class Graph:
    def __init__(self):
        self.__nodes = dict()

    def addNode(self, node):
        self.__nodes[node.getPos()] = node

    def getNode(self, pos):
        return self.__nodes.get(pos, None)

    # Add an edge from node1 to node2
    def addEdge(self, node1, node2):
        if not (node1 in self.__nodes and node2 in self.__nodes):
            node1.addNeighbor(node2)

class Map2D:
    def __init__(self, filePath):
        self.__graph = Graph()
        self.__startNode = None
        self.__endNode = None

        with open(filePath) as inFile:
            lines = inFile.readlines()

        self.__dims = (len(lines), len(lines[0])-1)
        
        # add Nodes to graph
        for i in range(len(lines)):
            nodeLine = list()
            for j in range(len(lines[i])):
                #each character is a node
                char = lines[i][j]
                if "\n" == char:
                    continue
                node = Node(char,i,j)
                # "#" for non traversable nodes
                if "#" == char:
                    node.setTraversability(False)
                else:
                    node.setTraversability(True)
                    if "S" == char:
                        self.__startNode = node
                    elif "E" == char:
                        self.__endNode = node
                self.__graph.addNode(node)

        if self.__startNode == None or self.__endNode == None:
            sys.exit("No start or no end in map")

        # for each node, for each neighbor, add edge between neighbor and node
        for i in range(self.__dims[0]):
            for j in range(self.__dims[1]):
                currentNode = self.__graph.getNode((i,j))
                if self.__isValid(i+1,j):
                    self.__graph.addEdge(currentNode,self.__graph.getNode((i+1,j)))
                if self.__isValid(i-1,j):
                    self.__graph.addEdge(currentNode,self.__graph.getNode((i-1,j)))
                if self.__isValid(i,j+1):
                    self.__graph.addEdge(currentNode,self.__graph.getNode((i,j+1)))
                if self.__isValid(i,j-1):
                    self.__graph.addEdge(currentNode,self.__graph.getNode((i,j-1)))

    def printMap(self):
        if None == self.__dims:
            print("ERROR: map not initialized")
            return

        for i in range(self.__dims[0]):
            for j in range(self.__dims[1]):
                print(self.__graph.getNode((i,j)).getKey(), end="")
            print()

    def __isValid(self, i, j):
        node = self.__graph.getNode((i,j))
        # if i,j are valid indices and the node is traversable
        if node != None and node.isTraversable():
            return True
        return False

    def getStart(self):
        return self.__startNode

    def getEnd(self):
        return self.__endNode

class AStar:
    def __init__(self, inMap):
        # holds list of nodes from which the next node is chosen
        self.__openList = queue.PriorityQueue()

        # put start node in openList
        start = inMap.getStart()
        start.setDistFromStart(0)
        start.setPriority(self.__heuristic(start, inMap.getEnd()))
        start.setParent(None)
        self.__openList.put((start.getPriority(),start))

        self.__map = inMap

        # list of visited nodes
        self.__closedList = dict()

    # setting this to return 0, will comvert this to Dijkistra's Algorithm
    # manhattan distance
    def __heuristic(self, node, targetNode):
        i1,j1 = node.getPos()
        i2,j2 = targetNode.getPos()
        return abs(i1-i2)+abs(j1-j2)

    def findPath(self):
        while True:
            # OL is empty implies, no route, stop
            if self.__openList.empty():
                print("No route!")
                return

            # get highest priority element from OL
            priority, currentNode = self.__openList.get()

            # if we reached end node, follow parents to get path
            if currentNode == self.__map.getEnd():
                # follow parents till startNode
                while currentNode != self.__map.getStart():
                    # set "*" for visual output
                    currentNode.setKey("*")
                    currentNode = currentNode.getParent()
                currentNode.setKey("*")
                return

            # unweighted graph should add 1, otherwise replace 1 with edge weight
            newLevel = currentNode.getDistFromStart() + 1

            # add neighbors to OL
            for neighbor in currentNode.getNeighbors():
                # if it has already been visited
                if self.__closedList.get(neighbor.getPos(), None) != None:
                    continue

                # if it is in OL but the new path is longer than the already existing one
                if "O" == neighbor.getKey() and newLevel >= neighbor.getDistFromStart():
                    continue

                # add neighbor to OL
                neighbor.setKey("O")
                neighbor.setParent(currentNode)
                neighbor.setDistFromStart(newLevel)
                neighbor.setPriority(neighbor.getDistFromStart() + self.__heuristic(neighbor,self.__map.getEnd()))
                self.__openList.put((neighbor.getPriority(), neighbor))
            # mark node as visited
            self.__closedList[currentNode.getPos()] = True
            currentNode.setKey("C")

            # un-comment following 2 lines to see how algorithm works
            # self.__map.printMap()
            # input()

    def getMap(self):
        return self.__map

if "__main__" == __name__:
    inpMap = Map2D("./data/graph.in")
    print("Initial Map:")
    inpMap.printMap()
    aStar = AStar(inpMap)
    aStar.findPath()
    print("Map with route(*), unvisited nodes(-/O/#), visited nodes(C)")
    aStar.getMap().printMap()