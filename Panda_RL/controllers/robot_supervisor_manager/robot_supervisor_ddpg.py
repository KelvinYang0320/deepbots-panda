from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalizeToRange

from gym.spaces import Box, Discrete
import numpy as np
from ArmUtil import Func, ToArmCoord
import time
class PandaRobotSupervisor(RobotSupervisor):
    """
    Observation:
        Type: Box(10)
        Num	Observation                Min(rad)      Max(rad)
        0	Target x                   -Inf           Inf
        1	Target y                   -Inf           Inf
        2	Target z                   -Inf           Inf
        3	Position Sensor on A1      -2.8972        2.8972
        4	Position Sensor on A2      -1.7628        1.7628
        5	Position Sensor on A3      -2.8972        2.8972
        6	Position Sensor on A4      -3.0718       -0.0698
        7	Position Sensor on A5      -2.8972        2.8972
        8   Position Sensor on A6      -0.0175        3.7525
        9	Position Sensor on A7      -2.8972        2.8972
        
    Actions:
        Type: Discrete(2)
        Num	  Action
        0	  
        ...
        3^7-1 
    Reward:
        Reward is -L2 norm for every step taken, including the termination step
    Starting State:
        ...
    Episode Termination:
        ...
    """

    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
                                     high=np.array([np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]),
                                     dtype=np.float64)
        self.action_space = Discrete(7)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.positionSensorList = Func.get_All_positionSensors(self, self.timestep)
        # self.positionSensor.enable(self.timestep)
        self.target = self.getFromDef("TARGET%s"%(np.random.randint(1, 10, 1)[0]))

        self.setup_motors()

        # Set up misc
        self.stepsPerEpisode = 300  # How many steps to run each episode (changing this messes up the solved condition)
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.motorVelocity = 2.5
        self.deltaAngle = 0.05
        
        # Set these for ensure that the robot stops moving
        self.motorPositionArr = np.zeros(7)
        self.motorPositionArr_target = np.zeros(7)
        self.distance = float("inf")
        self.endEffector = self.getFromDef("endEffector")
        
    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the CartPole problem.
        All values apart are gathered here from the robot and poleEndpoint objects.
        All values are normalized appropriately to [-1, 1], according to their original ranges.

        :return: Observation: [cartPosition, cartVelocity, poleAngle, poleTipVelocity]
        :rtype: list
        """
        # process of negotiation
        prec = 0.0001
        # print("err:",np.absolute(np.array(self.motorPositionArr)-np.array(self.motorPositionArr_target)))
        err = np.absolute(np.array(self.motorPositionArr)-np.array(self.motorPositionArr_target)) < prec
        if not np.all(err):
            return ["StillMoving"]
        # ----------------------
        
        targetPosition = ToArmCoord.convert(self.target.getPosition())
        message = [i for i in targetPosition]
        message.extend([i for i in self.motorPositionArr])
        return message

    def get_reward(self, action):
        """
        Reward is +1 for each step taken, including the termination step.

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: Always 1
        :rtype: int
        """
        targetPosition = self.target.getPosition()
        targetPosition = ToArmCoord.convert(targetPosition)
        endEffectorPosition = self.endEffector.getPosition()
        endEffectorPosition = ToArmCoord.convert(endEffectorPosition)
        self.distance = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
        reward = -self.distance
        if self.distance < 0.01:
            reward = reward + 1.5
        elif self.distance < 0.015:
            reward = reward + 1.0
        elif self.distance < 0.03:
            reward = reward + 0.5
        return reward

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is on the
        arena's edges.

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if(self.distance < 0.005):
            done = True
        else:
            done = False
        return done

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episode scores average value
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]
    def motorToRange(self, motorPosition, i):
        if(i==0):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i==1):
            motorPosition = np.clip(motorPosition, -1.7628, 1.7628)
        elif(i==2):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i==3):
            motorPosition = np.clip(motorPosition, -3.0718, -0.0698)
        elif(i==4):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i==5):
            motorPosition = np.clip(motorPosition, -0.0175, 3.7525)
        elif(i==6):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        else:
            pass
        return motorPosition
    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        It contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motorSpeed value is applied to the wheels.

        :param action: The list that contains the action integer
        :type action: list of int
        """
        # ignore this action and keep moving
        if action[0]==-1 and len(action)==1:
            for i in range(7):
                motorPosition = self.positionSensorList[i].getValue()
                self.motorPositionArr[i]=motorPosition
                self.motorList[i].setVelocity(2.5)
                self.motorList[i].setPosition(self.motorPositionArr_target[i])
            return
        setActionList = [i for i in action]
        self.motorPositionArr = np.array(Func.getValue(self.positionSensorList))
        for i in range(7):
            action = setActionList[i]
            motorPosition = self.motorPositionArr[i] + action
            motorPosition = self.motorToRange(motorPosition, i)
            self.motorList[i].setVelocity(self.motorVelocity)
            self.motorList[i].setPosition(motorPosition)
            self.motorPositionArr_target[i]=motorPosition # motorPositionArr_target Update
        # print(self.motorPositionArr, self.motorPositionArr_target)

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motorList = Func.get_All_motors(self)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")
