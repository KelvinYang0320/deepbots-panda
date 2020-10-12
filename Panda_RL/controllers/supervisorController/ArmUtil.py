from controller import PositionSensor
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
	"""
	Convert from world coordinate (x, y, z)
	to arm coordinate (x, -z, y)
	"""
	@staticmethod
	def getValue(positionSensorList):
		psValue = []
		for i in positionSensorList:
			psValue.append(i.getValue())
		return psValue