from numpy import convolve, ones, mean, random
import numpy as np
from robot_supervisor_ddpg import PandaRobotSupervisor
from agent.ddpg import DDPGAgent

from Constants import EPISODE_LIMIT, STEPS_PER_EPISODE, SAVE_MODELS_PERIOD, MOTOR_VELOCITY
from agent.ppo import PPOAgent, Transition
from ikpy.chain import Chain
from ArmUtil import ToArmCoord

import time
def ikpy_panda_balance(armChain, env):
    ikResults = armChain.inverse_kinematics(
        target_position=ToArmCoord.convert(env.target.getPosition()), 
        target_orientation=[0,0,1],
        orientation_mode="X",
        initial_position=np.insert(np.insert(env.motorPositionArr, 0, 0), 8, 0))

    action = ikResults[1:8]
    for i in range(7):
        motorPosition = action[i]
        motorPosition = env.motorToRange(motorPosition, i)
        env.motorList[i].setVelocity(MOTOR_VELOCITY)
        env.motorList[i].setPosition(motorPosition)
        env.motorPositionArr_target[i]=motorPosition # Update motorPositionArr_target 
    for i in range(7):
        env.motorPositionArr[i] = env.positionSensorList[i].getValue()
    prec = 0.0001
    err = np.absolute(np.array(env.motorPositionArr)-np.array(env.motorPositionArr_target)) < prec
    if not np.all(err):
        return False
    else:
        return True
def run(load_path):
    armChain = Chain.from_urdf_file("./panda_with_bound.URDF")
    # Initialize supervisor object
    env = PandaRobotSupervisor()
    
    # agent_balance = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]-3], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.shape[0], load_path=load_path+'ddpg_balance/') 
    agent_goal = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.shape[0], load_path=load_path+'ddpg_goal/') 
    
    episodeCount = 0 
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        state = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0
        actionProbs = []

        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0])) # Select one of the targets
        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            print(step)
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            
            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            act = agent_goal.choose_action_test(state)

            newState, reward, done, info = env.step(act*0.032)

            # process of negotiation
            while(newState==["StillMoving"]):
                newState, reward, done, info = env.step([-1])
            

            env.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            if done or step==STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                
                if episodeCount%SAVE_MODELS_PERIOD==0:
                    pass
                solved = env.solved()  # Check whether the task is solved
                break

            state = newState # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter

        

        for step in range(100):
            print(step)
            pos = ToArmCoord.convert(env.target.getPosition())
            pos[2] = pos[2]-0.44
            ikResults = armChain.inverse_kinematics(
                target_position=pos,
                target_orientation=[0,0,1],
                orientation_mode="X",
                initial_position=np.insert(np.insert(env.motorPositionArr, 0, 0), 8, 0))
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            
            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            act = agent_goal.choose_action_test(state)

            newState, reward, done, info = env.step(ikResults)

            # process of negotiation
            while(newState==["StillMoving"]):
                newState, reward, done, info = env.step([-1])
                print("waiting...")
            if newState!=["StillMoving"]:
                break
            

            env.episodeScore += reward  # Accumulate episode reward

            state = newState # state for next step is current step's newState

        

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episodeScore = 0
    step = 0
    env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0])) # Select one of the targets
    while True:
        act = agent.choose_action_test(state)
        state, reward, done, _ = env.step(act*0.032)
        # process of negotiation
        while(state==["StillMoving"]):
            state, reward, done, info = env.step([-1])
        
        env.episodeScore += reward  # Accumulate episode reward
        step = step + 1
        if done or step==STEPS_PER_EPISODE-1:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            state = env.reset()
            step = 0
            env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0]))
        
