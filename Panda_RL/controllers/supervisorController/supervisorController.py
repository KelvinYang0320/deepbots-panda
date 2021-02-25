import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from agent.PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange
from ArmUtil import ToArmCoord, PSFunc
import time
# from ikpy.chain import Chain
# from ikpy.link import OriginLink, URDFLink
class PandaSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()  
		self.observationSpace = 10  # The agent has 7 inputs 
		self.actionSpace = 2187-1 # The agent can perform 3^7-1 actions

		self.robot = None
		self.target = None
		self.target = self.supervisor.getFromDef("TARGET")
		self.endEffector = None 
		# self.armChain = Chain.from_urdf_file("panda_with_bound.URDF")
		self.respawnRobot()
		self.messageReceived = None	 # Variable to save the messages received from the robot
		
		self.episodeCount = 0  # Episode counter
		self.episodeLimit = 15000  # Max number of episodes allowed
		self.stepsPerEpisode = 300  # Max number of steps per episode
		self.episodeScore = 0  # Score accumulated during an episode
		self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
		self.doneTotal = 0

		self.distance = float("inf")

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
		
		# print("test:",self.endEffector.getPosition())
	def get_observations(self): # [motorPosition, targetPosition, endEffectorPosition, L2norm(targetPosition vs endEffectorPosition)]
		# Update self.messageReceived received from robot, which contains motor position
		self.messageReceived = self.handle_receiver()
		# print("[Get a Message]")
		# print("[Obs]", self.messageReceived)
		if self.messageReceived is None:
			return ["StillMoving"]
		if self.messageReceived is not None:
			# print("[Get a Message]:")
			if(len(self.messageReceived)==1):
				returnObservation = ["StillMoving"]
				return returnObservation
			motorPosition = [ float(i)  for i in self.messageReceived]
		else:
			# Method is called before self.messageReceived is initialized
			motorPosition = [0.0 for _ in range(7)]
			motorPosition[3] = -0.0698
		
		# get TARGET posion
		targetPosition = self.target.getPosition()
		targetPosition = ToArmCoord.convert(targetPosition) # transfer to arm coordinate system
		
		# convert to a single list
		returnObservation = [*motorPosition, *targetPosition]
		return returnObservation
		
	def get_reward(self, action=None):

		# print("[R]", self.messageReceived)
		if self.messageReceived is not None:
			# print("[Get a Message]:")
			if(len(self.messageReceived)==1):
				returnObservation = ["StillMoving"]
				return 0
		
		# get TARGET posion
		targetPosition = self.target.getPosition()
		targetPosition = ToArmCoord.convert(targetPosition) # transfer to arm coordinate system

		if self.messageReceived is not None:
			# get end-effort position and transfer to arm coordinate system
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) 
			# compute L2 norm
			self.distance = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		else:
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
		
		reward = -self.distance
		
		if self.distance < 0.01: 
			reward = reward + 1.5
		elif self.distance < 0.015:
			reward = reward + 1.0
		elif self.distance < 0.03:
			reward = reward + 0.5
		
		return reward 
	
	def is_done(self):

		if(self.distance < 0.005):
			done = True
		else:
			done = False

		return done
	
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
agent.load('')
solved = False

cnt_veryClose = 0
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episodeCount < supervisor.episodeLimit:
	observation = supervisor.reset()  # Reset robot and get starting observation
	supervisor.episodeScore = 0
	cnt_veryClose = 0
	print("===episodeCount:", supervisor.episodeCount,"===")
	for step in range(supervisor.stepsPerEpisode):
		# print("===step:", step,"===")
		# In training mode the agent samples from the probability distribution, naturally implementing exploration
		selectedAction, actionProb = agent.work(observation, type_="selectAction")
		
		# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
		# the done condition
		# print("step:", step, "| selectedAction:", selectedAction)
		newObservation, reward, done, info = supervisor.step([selectedAction])
		# start_t = time.time()
		while(newObservation==["StillMoving"]):
			newObservation, reward, done, info = supervisor.step([-1])
		# end_t = time.time()
		# print("time:",end_t-start_t)
		# print("Ready for next step!")
		# print("~~~~~~~~~~~~~~~~~~~~")
		
		# print(newObservation, reward, done, info)

		# Save the current state transition in agent's memory
		trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
		agent.storeTransition(trans)
		# print(supervisor.distance,"|", done)
		if done or step == supervisor.stepsPerEpisode-1:
			if(step==0):
				print("Warning: Step=0 but the task is done?")
				continue
			if done:
				print("Done!, Now training start...")
			if step == supervisor.stepsPerEpisode-1:
				print("Run through all steps! Now training start...")
			# Save the episode's score
			supervisor.episodeScoreList.append(supervisor.episodeScore)
			agent.trainStep(batchSize=step)
			solved = supervisor.solved()  # Check whether the task is solved
			agent.save('')
			break
		# print("reward:--------------------------------------===",reward)
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
