from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import numpy as np
from ArmUtil import Func
def motorToRange(motorPosition, i):
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
class PandaRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		# get all position sensors
		self.positionSensorList = Func.get_All_positionSensors(self.robot, self.get_timestep())
		# get all motor
		self.motorList = Func.get_All_motors(self.robot)

		# Set these for ensure that the robot stops moving
		self.motorPositionArr = np.zeros(7)
		self.motorPositionArr_target = np.zeros(7)

		# rotation for Each step
		self.deltaAngle = 0.05
		self.motorVelocity = 2.5

	def create_message(self):
		prec = 0.0001
		err = (np.array(self.motorPositionArr)-np.array(self.motorPositionArr_target)) < prec

		if not np.all(err):
			message = ["StillMoving"]
		else:
			# Read the sensor value, convert to string and save it in a list
			message = [str(i) for i in self.motorPositionArr]
		return message
	
	def use_message_data(self, message):
		#print("robot get this message: ", message)
		code = int(message[0])
		
		# ignore this action and keep moving
		if code==-1:
			for i in range(7):
				motorPosition = self.positionSensorList[i].getValue()
				self.motorPositionArr[i]=motorPosition
				self.motorList[i].setVelocity(2.5)
				self.motorList[i].setPosition(self.motorPositionArr_target[i]) 
			return
		
		setActionList = []
		# decoding action
		for i in range(7):
			setActionList.append(code%3)
			code = int(code/3)
		#print("decode message to action: ", setActionList)
		self.motorPositionArr = np.array(Func.getValue(self.positionSensorList))
		for i in range(7):
			action = setActionList[i]
			if action == 2:
				motorPosition = self.motorPositionArr[i]-self.deltaAngle
			elif action == 1:
				motorPosition = self.motorPositionArr[i]+self.deltaAngle
			else:
				motorPosition = self.motorPositionArr[i]
			motorPosition = motorToRange(motorPosition, i)
			self.motorList[i].setVelocity(self.motorVelocity)
			self.motorList[i].setPosition(motorPosition)
			self.motorPositionArr_target[i]=motorPosition
# Create the robot controller object and run it
robot_controller = PandaRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
