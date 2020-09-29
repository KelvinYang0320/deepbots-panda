import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange
from ArmUtil import ToArmCoord, PSFunc
# from ikpy.chain import Chain
# from ikpy.link import OriginLink, URDFLink
class PandaSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()  
		self.observationSpace = 14  # The agent has 7 inputs 
		self.actionSpace = 2187 # The agent can perform 3^7 actions

		self.robot = None
		self.target = None
		self.target = self.supervisor.getFromDef("TARGET")
		self.endEffector = None 
		# self.armChain = Chain.from_urdf_file("panda_with_bound.URDF")
		self.respawnRobot()
		self.messageReceived = None	 # Variable to save the messages received from the robot
		
		self.episodeCount = 0  # Episode counter
		self.episodeLimit = 15000  # Max number of episodes allowed
		self.stepsPerEpisode = 500  # Max number of steps per episode
		self.episodeScore = 0  # Score accumulated during an episode
		self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
		self.doneTotal = 0

		# for Computing the distance diff
		# targetPosition = ToArmCoord.convert(self.target.getPosition()) # transfer to arm coordinate system
		# tmp = [0.0 for _ in range(8)]
		# tmp[3] = -0.0698
		# endEffectorPosition = self.armChain.forward_kinematics(tmp)[0:3, 3] # This is already in arm coordinate.
		# endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
		# self.preL2norm = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		#------
	def respawnRobot(self):
		if self.robot is not None:
			# Despawn existing robot
			self.robot.remove()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
		childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
		childrenField.importMFNode(-2, "Robot.wbo")	 # Load robot from file and add to second-to-last position

		# Get the new robot 
		self.robot = self.supervisor.getFromDef("Franka_Emika_Panda")
		self.endEffector = self.supervisor.getFromDef("endEffector")

		targetPosition = ToArmCoord.convert(self.target.getPosition()) # transfer to arm coordinate system
		endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
		self.preL2norm = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		# print("test:",self.endEffector.getPosition())
	def get_observations(self): # [motorPosition, targetPosition, endEffectorPosition, L2norm(targetPosition vs endEffectorPosition)]
		# Update self.messageReceived received from robot, which contains motor position
		self.messageReceived = self.handle_receiver()
		if self.messageReceived is not None:
			print("yes")
			motorPosition = [float(self.messageReceived[0]), float(self.messageReceived[1]), float(self.messageReceived[2]), \
				float(self.messageReceived[3]), float(self.messageReceived[4]), float(self.messageReceived[5]), \
				float(self.messageReceived[6])]
		else:
			# Method is called before self.messageReceived is initialized
			motorPosition = [0.0 for _ in range(7)]
			motorPosition[3] = -0.0698
		
		# get TARGET posion
		targetPosition = self.target.getPosition()
		targetPosition = ToArmCoord.convert(targetPosition) # transfer to arm coordinate system
		
		if self.messageReceived is not None:
			# get end-effort position
			motorPosition_for_FK = motorPosition + [0.0]
			# endEffectorPosition = self.armChain.forward_kinematics(motorPosition_for_FK)[0:3, 3] # This is already in arm coordinate.
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
			# compute L2 norm
			# print("[Debug tartgetPosition]:",targetPosition)
			# print("[Debug endEffectorPosition]:",endEffectorPosition)
			L2norm = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		else:
			# tmp = [0.0 for _ in range(8)]
			# tmp[3] = -0.0698
			# endEffectorPosition = self.armChain.forward_kinematics(tmp)[0:3, 3] # This is already in arm coordinate.
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
			L2norm = self.preL2norm
		
		# convert to a single list
		returnObservation = [*motorPosition, *targetPosition, *endEffectorPosition, L2norm]
		return returnObservation
		
	def get_reward(self, action=None):
		return 0 # I implement in the comment:'compute reward here' 
	
	def is_done(self):
		return False
	
	def solved(self):
		if len(self.episodeScoreList) > 1000:  # Over 100 trials thus far
			if np.mean(self.episodeScoreList[-100:]) > 1000:  # Last 100 episodes' scores average value
				return True
		return False
		
	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		self.messageReceived = None
		return [0.0 for _ in range(self.observationSpace)]
		
	def get_info(self):
		return "I'm trying to reach that red ball!"
		
supervisor = PandaSupervisor()
agent = PPOAgent(supervisor.observationSpace, supervisor.actionSpace, use_cuda=True) #add use_cuda

solved = False

cnt_veryClose = 0
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episodeCount < supervisor.episodeLimit:
	observation = supervisor.reset()  # Reset robot and get starting observation
	supervisor.episodeScore = 0
	cnt_veryClose = 0
	for step in range(supervisor.stepsPerEpisode):
		print("step: ", step)
		# In training mode the agent samples from the probability distribution, naturally implementing exploration
		selectedAction, actionProb = agent.work(observation, type_="selectAction")
		
		# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
		# the done condition
		newObservation, reward, done, info = supervisor.step([selectedAction])
		print("L2",newObservation[-1])
		# compute done here
		if newObservation[-1] <= 0.01:
			cnt_veryClose += 1
		if cnt_veryClose >= 50 or step==supervisor.stepsPerEpisode-1:
			done = True
			supervisor.preL2norm=0.4
		# compute reward here
		## do not get too close to the limit value 
		# [-2.897, 2.897], [-1.763, 1.763], [-2.8973, 2.8973], [-3.072, -0.0698]
		# [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.897, 2.897]
		if newObservation[0]-(-2.897)<0.05 or 2.897-newObservation[0]<0.05 or\
			newObservation[1]-(-1.763)<0.05 or 1.763-newObservation[1]<0.05 or\
			newObservation[2]-(-2.8973)<0.05 or 2.8973-newObservation[2]<0.05 or\
			newObservation[3]-(-3.072)<0.05 or -0.0697976-newObservation[3]<0.05 or\
			newObservation[4]-(-2.8973)<0.05 or 2.8973-newObservation[4]<0.05 or\
			newObservation[5]-(-0.0175)<0.05 or 3.7525-newObservation[5]<0.05 or\
			newObservation[6]-(-2.897)<0.05 or 2.897-newObservation[6]<0.05:
			reward = -1 # if on of the motors on the limit, reward = -2
		else:
			if(newObservation[-1]<0.01):
				reward = 10 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode) 
			elif(newObservation[-1]<0.05):
				reward = 5 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode)
			elif(newObservation[-1]<0.1):
				reward = 1 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode)
			else:
				reward = -(newObservation[-1]-supervisor.preL2norm) # negative reward
			supervisor.preL2norm = newObservation[-1]
 
		print("reward: ",reward)
		print("L2norm: ", newObservation[-1])
		print("tarPosition(trans): ", newObservation[7:10])
		print("endPosition: ", newObservation[10:13])
		#print("endPosition(trans): ", ToArmCoord.convert(newObservation[10:13]))
		# ------compute reward end------
		# Save the current state transition in agent's memory
		trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
		agent.storeTransition(trans)

		
		if done:
			if(step==0):
				print("0 Step but done?")
				continue
			print("done gogo")
			# Save the episode's score
			supervisor.episodeScoreList.append(supervisor.episodeScore)
			agent.trainStep(batchSize=step)
			solved = supervisor.solved()  # Check whether the task is solved
			agent.save('')
			break
		
		supervisor.episodeScore += reward  # Accumulate episode reward
		observation = newObservation  # observation for next step is current step's newObservation
		
	fp = open("Episode-score.txt","a")
	fp.write(str(supervisor.episodeScore)+'\n')
	fp.close()
	print("Episode #", supervisor.episodeCount, "score:", supervisor.episodeScore)
	supervisor.episodeCount += 1  # Increment episode counter

if not solved:
	print("Task is not solved, deploying agent for testing...")
elif solved:
	print("Task is solved, deploying agent for testing...")
	
observation = supervisor.reset()

while True:
	selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
	observation, _, _, _ = supervisor.step([selectedAction])
