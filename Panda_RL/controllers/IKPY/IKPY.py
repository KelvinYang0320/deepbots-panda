"""IKPY_0 controller."""
from controller import Supervisor
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ArmUtil import ToArmCoord, PSFunc, Func
from controller import PositionSensor
import numpy as np

armChain = Chain.from_urdf_file("./panda_with_bound.URDF")

supervisor = Supervisor()
timestep = int(16*supervisor.getBasicTimeStep())
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
motorList = Func.get_All_motors(supervisor)
#連接馬達
motorList = []
for i in range(7):
    motorName = 'motor' + str(i + 1)
    motor = supervisor.getDevice(motorName)
    motor.setVelocity(1.0)
    motorList.append(motor)

_motor = supervisor.getDevice('finger motor L')
_motor.setVelocity(0.1)
motorList.append(_motor)
_motor = supervisor.getDevice('finger motor R')
_motor.setVelocity(0.1)
motorList.append(_motor)

target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))
arm = supervisor.getFromDef('Franka_Emika_Panda')

i = 0
ik_mode = 5
rst_flag = 0
print("[IKPY start]")
while supervisor.step(timestep) != -1:
    #keyboard Control
    key=keyboard.getKey()
    if (key==ord('A')):
        motorList[7].setVelocity(0.01)
        motorList[8].setVelocity(0.01)
        print('+finger motor')
    elif (key==ord('S')):
        motorList[7].setVelocity(-0.01)
        motorList[8].setVelocity(-0.01)
        print('-finger motor')
    elif (key==ord('D')):
        print("[Info] Changing a goal...")
        target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))
        targetPosition = target.getPosition()
        print("Target Position:", targetPosition)
    elif (key==ord('0')):
        ik_mode = 0
        print('[Info] Switching to ik mode 0')
    elif (key==ord('1')):
        ik_mode = 1
        print('[Info] Switching to ik mode 1')
    elif (key==ord('2')):
        ik_mode = 2
        print('[Info] Switching to ik mode 2')
    elif (key==ord('3')):
        ik_mode = 3
        print('[Info] Switching to ik mode 3')
    elif (key==ord('4')):
        ik_mode = 4
        print('[Info] Switching to ik mode 4')
    elif (key==ord('5')):
        ik_mode = 5
        print('[Info] Switching to ik mode 5')
    elif (key==ord('P')):
        print(PSFunc.getValue(positionSensorList))
    elif (key==ord('R')):
        rst_flag = 0 if rst_flag else 1 
        print('[Info] Reseting')
    else:
        motorList[7].setVelocity(0)
        motorList[8].setVelocity(0)
    
    
    targetPosition = target.getPosition()
    
    armPosition = arm.getPosition()
    
    targetPosition = ToArmCoord.convert(targetPosition)
    armPosition = ToArmCoord.convert(armPosition)

    
    x = targetPosition[0] - armPosition[0]
    y = targetPosition[1] - armPosition[1]
    z = targetPosition[2] - armPosition[2]

    # get target
    frame_target = np.eye(4)
    frame_target[:3, 3] = targetPosition
    
    # get psValue
    psValue = PSFunc.getValue(positionSensorList)
    psValue.append(0)
    psValue = np.array(psValue)
    psValue = np.insert(psValue, 0, 0)
    
    if (ik_mode == 1):
        ikResults = armChain.inverse_kinematics(targetPosition)
    elif (ik_mode == 2):
        ikResults = armChain.inverse_kinematics(
            targetPosition, 
            initial_position=psValue)
    elif (ik_mode == 3):
        ikResults = armChain.inverse_kinematics(
            target_position=targetPosition, 
            target_orientation=[0,0,-1],
            orientation_mode="Z")
    elif (ik_mode == 4):
        ikResults = armChain.inverse_kinematics(
            target_position=targetPosition, 
            target_orientation=[0,0,-1],
            orientation_mode="Z",
            initial_position=psValue)
    elif (ik_mode == 5):
        ikResults = armChain.inverse_kinematics(
            target_position=targetPosition, 
            target_orientation=[0,0,1],
            orientation_mode="X",
            initial_position=psValue)
    else:
        ikResults = armChain.inverse_kinematics_frame(
            target=frame_target, 
            initial_position=psValue)
    
    if rst_flag:
        for i in range(7):
            motorList[i].setPosition(0.0)
            motorList[i].setVelocity(1.0)
        continue
    for i in range(7):
        motorList[i].setPosition(ikResults[i+1])
        motorList[i].setVelocity(1.0)