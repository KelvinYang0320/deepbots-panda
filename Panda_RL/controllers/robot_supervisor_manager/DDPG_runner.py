from numpy import convolve, ones, mean, random

from robot_supervisor_ddpg import PandaRobotSupervisor
from agent.ddpg import DDPGAgent


def run():
    # Initialize supervisor object
    env = PandaRobotSupervisor()

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1509.02971).
    # We pass 10 as numberOfInputs and 7 as numberOfOutputs, taken from the gym spaces
    agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.n) 
              
    agent.load_models() # Load the pretrained model
    episodeCount = 0 
    episodeLimit = 50000
    solved = False  # Whether the solved requirement is met
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0
        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0]))
        # Inner loop is the episode loop
        for step in range(env.stepsPerEpisode):
            # print("===step:", step,"===")
            # In training mode the agent returns the action plus OU noise for exploration
            act = agent.choose_action(state)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # the done condition
            newState, reward, done, info = env.step(act*0.032)
            # process of negotiation
            while(newState==["StillMoving"]):
                newState, reward, done, info = env.step([-1])
            
            # Save the current state transition in agent's memory
            agent.remember(state, act, reward, newState, int(done))

            env.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            if done or step==env.stepsPerEpisode-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.learn()
                solved = env.solved()  # Check whether the task is solved
                break

            state = newState # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter
        
    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episodeScore = 0
    step = 0
    env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0]))
    while True:
        act = agent.choose_action_test(state)
        state, reward, done, _ = env.step(act*0.032)
        # process of negotiation
        while(state==["StillMoving"]):
            state, reward, done, info = env.step([-1])
        
        env.episodeScore += reward  # Accumulate episode reward
        step = step + 1
        if done or step==env.stepsPerEpisode-1:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            state = env.reset()
            step = 0
            env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0]))
        
