"""IKPY_0 controller."""
from controller import Supervisor
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ArmUtil import ToArmCoord, PSFunc
import PandaController
from controller import PositionSensor, Motor
import numpy as np

armChain = Chain.from_urdf_file("panda_with_bound.URDF")

supervisor = Supervisor()
timestep = int(4*supervisor.getBasicTimeStep())
#keyboard control
robot = supervisor
keyboard=robot.getKeyboard()
keyboard.enable(1)
#-----------
#position Sensor
positionSensorList = []
for i in range(7):
    psName = 'positionSensor' + str(i + 1)
    ps=PositionSensor(psName)
    ps.enable(1)
    positionSensorList.append(ps)
#-----------
motorList = []
#連接馬達
for i in range(7):
    motorName = 'motor' + str(i + 1)
    motor = Motor(motorName)
    motor.setVelocity(1.0)
    motorList.append(motor)
#_motor = supervisor.getMotor('finger motor L')
_motor = Motor('finger motor L')
_motor.setVelocity(0.1)
motorList.append(_motor)
_motor = supervisor.getMotor('finger motor R')
_motor.setVelocity(0.1)
motorList.append(_motor)

target = supervisor.getFromDef('TARGET')
arm = supervisor.getFromDef('Franka_Emika_Panda')

fingertip = supervisor.getFromDef('fingerL')
joint12 = supervisor.getFromDef('joint12')

targetPosition = target.getPosition()
armPosition = arm.getPosition()

targetPosition = ToArmCoord.convert(targetPosition)
armPosition = ToArmCoord.convert(armPosition)

x = targetPosition[0] - armPosition[0]
y = targetPosition[1] - armPosition[1]
z = targetPosition[2] - armPosition[2]
relativeTargetPosition = [x,y,z]

#######################################
# 使用方法

# 1. 移動到目標點
controller = PandaController.PandaController(motorList, positionSensorList)
ActionList = [PandaController.IKReachTarget(relativeTargetPosition),
              #PandaController.IKReachTarget([x+0.2,y,z]),
              #PandaController.IKReachTarget([x-0.2,y,z])
             ]
script = PandaController.Script(ActionList)
controller.assign(script=script)

while True:
    break ## 砍掉這裡
    if (supervisor.step(timestep) == -1):
        break

    if (controller.done() == False):
        controller.step(direction_list=[0.1]*7)
        

# 2. 各關節正轉逆轉
controller = PandaController.PandaController(motorList, positionSensorList)
actuate_action = PandaController.Actuate([
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973)
],
[
    abs(-2.8973 - 2.8973),
    abs(-1.7628 - 1.7628),
    abs(-2.8973 - 2.8973),
    abs(-3.0718 - -0.0698),
    abs(-2.8973 - 2.8973),
    abs(-0.0175 - 3.7525),
    abs(-2.8973 - 2.8973)
],
step_count=1)
controller.assign(action=actuate_action)

while True:
    break ## 砍掉這裡
    if (supervisor.step(timestep) == -1):
        break
    if (controller.done() == False):
        controller.step(direction_list=[0.1]*7, 
            psValue=PSFunc.getValue(positionSensorList))
        print("step")
    else:
        print("done")


# 3. 移動到指定點 + 各關節正轉逆轉的腳本
controller = PandaController.PandaController(motorList, positionSensorList)
actuate_action = PandaController.Actuate(
    [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973)
    ],
    [
        abs(-2.8973 - 2.8973),
        abs(-1.7628 - 1.7628),
        abs(-2.8973 - 2.8973),
        abs(-3.0718 - -0.0698),
        abs(-2.8973 - 2.8973),
        abs(-0.0175 - 3.7525),
        abs(-2.8973 - 2.8973)
    ],
    step_count=3,
    direction_list=[0.1]*7
)
pause_0 = PandaController.Pause(10)

ActionList = [PandaController.IKReachTarget(relativeTargetPosition),
              pause_0,
              actuate_action]
script = PandaController.Script(ActionList)
controller.assign(script=script)

while True:
#    break  ## 砍掉這裡
    if (supervisor.step(timestep) == -1):
        break
        
    if (controller.done() == False):
        controller.step()


