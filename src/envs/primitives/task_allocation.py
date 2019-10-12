import numpy as np
import networkx as nx


class MRTA():
    # Author: Payam Ghassemi, payamgha@buffalo.edu
    # Sep 9, 2019
    # Copyright 2019 Payam Ghassemi
    """
    A class to handle Multi-robot Task Allocations.

    .. todo:: Optimize/learn weight function.
    """
    def __init__(self, vel=2, safeBattery=0.0):
        """Constructor of the class, initilize robots' parameters.

        Parameters
        ----------
        vel : int, optional
            Average velocity of robot, by default 2
        safeBattery : float, optional
            Min. battery to land (it is only set non-zero for UAVs)
            by default 0
        """
        self.safeBattery = safeBattery
        self.vel = vel
        self.insSep = 'i'

    def allocateRobots(self, robotInfo, groupInfo):  # noqa: C901
        """Allocate robots to given groups.

        Parameters
        ----------
        robotInfo : array, Nr by 3; Nr: number of robots
            Robots' information; row contains [pos-x, pos-y, Battery Level]
        groupInfo : array, Ng by 6; Ng: number of groups
            Groups' information; each row contains
            [c-x, c-y, # of robots, Priority, Min. Battery, Deadline Time]

        Returns
        -------
        array, Nr by 1
            Robot-group allocation; row-i is corresponding to robot-i
            and it contains the group that robot-i is allocated to.

        >>> from mrta import mrta
        >>> Nr = 15
        >>> robotInfo = np.random.rand(Nr,3)
        >>> Ng = 4
        >>> groupInfo = np.random.rand(Ng,6)
        >>> for i in range(Ng):
        >>>     groupInfo[i,2] = np.random.randint(1,3)
        >>> mrta = mrta()
        >>> robotGroupAlloc = mrta.allocateRobots(robotInfo, groupInfo)
        """

        nRobot = np.shape(robotInfo)[0]
        nGroup = np.shape(groupInfo)[0]
        robotNodes = np.arange(nRobot) + 1
        groupNodes = []

        insSep = self.insSep
        for iGroup in range(nGroup):
            nInstance = int(groupInfo[iGroup, 2])
            for j in range(nInstance):
                groupNodes.append('g' + str(iGroup + 1) + insSep + str(j + 1))

        B = nx.Graph()
        B.add_nodes_from(robotNodes, bipartite=0)
        B.add_nodes_from(groupNodes, bipartite=1)

        safeBattery = self.safeBattery
        vel = self.vel
        nGroupNodes = len(groupNodes)

        for iRobot in range(nRobot):
            robotBattery = robotInfo[iRobot, 2]
            if (robotBattery >= safeBattery):
                for iGroupNode in range(nGroupNodes):
                    groupNode = groupNodes[iGroupNode]
                    iGroup = self.getGroupId(groupNode) - 1
                    dist = np.linalg.norm(robotInfo[iRobot, :2] -
                                          groupInfo[iGroup, :2])
                    # p = groupInfo[iGroup, 3]
                    timeDeadline = groupInfo[iGroup, 5]
                    timeArrival = dist / vel
                    if (timeArrival <= timeDeadline):
                        weight = self.getWeight(robotInfo[iRobot, :],
                                                groupInfo[iGroup, :])
                        B.add_edge(robotNodes[iRobot],
                                   groupNode,
                                   weight=weight)

        robotGroup = np.zeros((nRobot, ))
        sol = nx.max_weight_matching(B, maxcardinality=True)

        for iGroupNode in range(nGroupNodes):
            groupNode = groupNodes[iGroupNode]
            if B.degree(groupNode) > 0:
                # print("sol: " + str(sol))
                for x, y in sol:
                    if x == groupNode:
                        dummySol = y
                        break
                    elif y == groupNode:
                        dummySol = x
                        break
                if dummySol > 0:
                    groupId = self.getGroupId(groupNode)
                    robotGroup[dummySol - 1] = groupId
        # ADDED
        return robotGroup.astype(int).tolist()

    def getWeight(self, robotInfo, groupInfo):
        """Retrieve group-id from group-nodes.

        Parameters
        ----------
        groupNode : string
            it has a format like 'gxxxixx', if
        you key insSep set at 'i'

        Returns
        -------
        int
            If 0, it's unallocated, otherwise it's the
        allocated group id
        """
        dist = np.linalg.norm(robotInfo[:2] - groupInfo[:2])
        p = groupInfo[3]
        if p == 0:
            p = 1

        return dist / p

    def getGroupId(self, groupNode):
        """Retrieve group-id from group-nodes.

        Parameters
        ----------
        groupNode : string
            Ut has a format like 'gxxxixx',
            if you key insSep set at 'i'

        Returns
        -------
        int
            group-id
        """
        groupId = ''
        for iGroupId in range(1, len(groupNode)):
            if groupNode[iGroupId] == self.insSep:
                break
            else:
                groupId = groupId + groupNode[iGroupId]

        return (int(groupId))
