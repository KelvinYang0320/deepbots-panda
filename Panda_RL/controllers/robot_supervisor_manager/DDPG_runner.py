from numpy import convolve, ones, mean, random

from robot_supervisor_ddpg import PandaRobotSupervisor
from agent.ddpg import DDPGAgent

from Constants import EPISODE_LIMIT, STEPS_PER_EPISODE, SAVE_MODELS_PERIOD
from agent.ppo import PPOAgent, Transition

def run(load_path):
    # Initialize supervisor object
    env = PandaRobotSupervisor()

    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    # We pass (10, ) as numberOfInputs and (7, ) as numberOfOutputs, taken from the gym spaces
    agent = PPOAgent(env.observation_space.shape[0], 2)
    
    agent_balance = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]-3], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.shape[0], load_path=load_path+'ddpg_balance/') 
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
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction, actionProb = agent.work(state, type_="selectAction")
            # Save the current selectedAction's probability
            actionProbs.append(actionProb)
            
            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            if selectedAction==0:
                act = agent_balance.choose_action_test(state[3:])
            else:
                act = agent_goal.choose_action_test(state)

            newState, reward, done, info = env.step(act*0.032)

            # process of negotiation
            while(newState==["StillMoving"]):
                newState, reward, done, info = env.step([-1])
            
            # Save the current state transition in agent's memory
            trans = Transition(state, selectedAction, actionProb, reward, newState)
            agent.storeTransition(trans)

            env.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            if done or step==STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.trainStep(batchSize=step + 1)
                
                if episodeCount%SAVE_MODELS_PERIOD==0:
                    agent.save('./tmp/ppo')
                solved = env.solved()  # Check whether the task is solved
                break

            state = newState # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter

    agent.save()
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
        
