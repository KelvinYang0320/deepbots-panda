"""IKPY_0 controller."""
from controller import Supervisor
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ArmUtil import ToArmCoord, PSFunc
import PandaController
from controller import PositionSensor, Motor
import numpy as np



armChain = Chain.from_urdf_file("panda_with_bound.urdf")

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
_motor = supervisor.getMotor('finger motor L')
_motor = Motor('finger motor L')
_motor.setVelocity(0.1)
motorList.append(_motor)
_motor = supervisor.getMotor('finger motor R')
_motor.setVelocity(0.1)
motorList.append(_motor)

motorList[7].setPosition(0.04)
motorList[8].setPosition(0.04)

Beaker = supervisor.getFromDef('Beaker')
target = supervisor.getFromDef('TARGET_')
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
              PandaController.IKReachTarget([x+0.2,y,z]),
              PandaController.IKReachTarget([x-0.2,y,z])
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
turnHor_action = PandaController.Actuate(
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
    step_count=30,
    direction_list=[0,0,0,0,0,0,0.25]
)
pause_0 = PandaController.Pause(20)

positionFront = relativeTargetPosition.copy()
positionFront[0] += 0.1
ActionList = [
    PandaController.IKReachTarget([x,y-0.25,z-0.05], orientation=[0,1,0], tolerance=0.01),
    PandaController.IKReachTarget(relativeTargetPosition, orientation=[0,1,0], lazyIK=False, tolerance=0.01),
    #turnHor_action,
    #PandaController.IKReachTarget(positionFront, orientation=[1,0,0]),
    #pause_0,
    #actuate_action
]
script = PandaController.Script(ActionList)
controller.assign(script=script)


while True:
#    break  ## 砍掉這裡
    if (supervisor.step(timestep) == -1):
        break
        
    if (controller.done() == False):
        print(controller.done())
        controller.step()
    elif(controller.done()!=False):
        motorList[7].setPosition(0.02)
        motorList[8].setPosition(0.02)
    #keyboard Control
    key=keyboard.getKey()
    if (key==ord('8')):
        motorList[7].setPosition(0.04)
        motorList[8].setPosition(0.04)
        print('+finger motor')
    elif (key==ord('I')):
        motorList[7].setPosition(0.02)
        motorList[8].setPosition(0.02)
        print('-finger motor')
    
    if (key==ord('D') and PSFunc.getValue(positionSensorList)[1]<2.897):
        motorList[1].setPosition(PSFunc.getValue(positionSensorList)[1]+0.03)
    elif(key==ord('C') and PSFunc.getValue(positionSensorList)[1]>-2.897):
        motorList[1].setPosition(PSFunc.getValue(positionSensorList)[1]-0.03)
    elif (key==ord('3') and PSFunc.getValue(positionSensorList)[2]<2.897):
        motorList[2].setPosition(PSFunc.getValue(positionSensorList)[2]+0.03)
    elif(key==ord('E') and PSFunc.getValue(positionSensorList)[2]>-2.897):
        motorList[2].setPosition(PSFunc.getValue(positionSensorList)[2]-0.03)
    elif (key==ord('4') and PSFunc.getValue(positionSensorList)[3]<2.897):
        motorList[3].setPosition(PSFunc.getValue(positionSensorList)[3]+0.03)
    elif(key==ord('R') and PSFunc.getValue(positionSensorList)[3]>-2.897):
        motorList[3].setPosition(PSFunc.getValue(positionSensorList)[3]-0.03)
    elif (key==ord('5') and PSFunc.getValue(positionSensorList)[4]<2.897):
        motorList[4].setPosition(PSFunc.getValue(positionSensorList)[4]+0.03)
    elif(key==ord('T') and PSFunc.getValue(positionSensorList)[4]>-2.897):
        motorList[4].setPosition(PSFunc.getValue(positionSensorList)[4]-0.03)
    elif (key==ord('6') and PSFunc.getValue(positionSensorList)[5]<2.897):
        motorList[5].setPosition(PSFunc.getValue(positionSensorList)[5]+0.03)
    elif(key==ord('Y') and PSFunc.getValue(positionSensorList)[5]>-2.897):
        motorList[5].setPosition(PSFunc.getValue(positionSensorList)[5]-0.03)
    else:
        print('no rotate arm')
    #hand rotate
    if (key==ord('7') and PSFunc.getValue(positionSensorList)[6]<2.897):
        motorList[6].setPosition(PSFunc.getValue(positionSensorList)[6]+0.03)
    elif(key==ord('U') and PSFunc.getValue(positionSensorList)[6]>-2.897):
        motorList[6].setPosition(PSFunc.getValue(positionSensorList)[6]-0.03)
    else:
        print('no rotate hand')
    print(Beaker.getOrientation())
    



