from controller import PositionSensor
import numpy as np
class ToArmCoord:
	"""
	Convert from world coordinate (x, y, z)
	to arm coordinate (x, -z, y)
	"""
	@staticmethod
	def convert(worldCoord):
		"""
		arg:
			worldCoord: [x, y, z]
				An array of 3 containing the 3 world coordinate.
		"""
		return [worldCoord[0], -worldCoord[2], worldCoord[1]]

class PSFunc:
	@staticmethod
	def getValue(positionSensorList):
		psValue = []
		for i in positionSensorList:
			psValue.append(i.getValue())
		return psValue
	
	@staticmethod
	def getInitialPosition(positionSensorList):
		psValue = []
		for i in positionSensorList:
			psValue.append(i.getValue())

		psValue.append(0)
		psValue = np.array(psValue)
		psValue = np.insert(psValue, 0, 0)
		return psValue