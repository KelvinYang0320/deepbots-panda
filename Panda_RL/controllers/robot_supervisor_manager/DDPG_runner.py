from numpy import convolve, ones, mean, random

from robot_supervisor_ddpg import PandaRobotSupervisor
from agent.ddpg import DDPGAgent
from utilities import plotData


def run():
    # Initialize supervisor object
    env = PandaRobotSupervisor()

    # The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).
    # We pass 4 as numberOfInputs and 2 as numberOfOutputs, taken from the gym spaces
    agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.n) 
              
    agent.load_models()
    episodeCount = 0 
    episodeLimit = 50000
    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later
    targetList = [3, 6, 9]
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0
        actionProbs = []  # This list holds the probability of each chosen action
        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET%s"%(random.randint(1, 10, 1)[0]))
        env.target = env.getFromDef("TARGET%s"%(targetList[episodeCount]))
        # Inner loop is the episode loop
        for step in range(env.stepsPerEpisode):
            # print("===step:", step,"===")
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            act = agent.choose_action(state)

            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # done condition
            newState, reward, done, info = env.step(act*0.032)
            # process of negotiation
            while(newState==["StillMoving"]):
                # print("[StillMoving]")
                newState, reward, done, info = env.step([-1])
            # ----------------------
            # Save the current state transition in agent's memory
            agent.remember(state, act, reward, newState, int(done))
            env.episodeScore += reward  # Accumulate episode reward
            if done or step==env.stepsPerEpisode-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.learn()
                solved = env.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        # avgActionProb = mean(actionProbs)
        # averageEpisodeActionProbs.append(avgActionProb)
        # print("Avg action prob:", avgActionProb)

        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    movingAvgN = 10
    plotData(convolve(env.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")
    plotData(convolve(averageEpisodeActionProbs, ones((movingAvgN,))/movingAvgN, mode='valid'),
             "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    state = env.reset()
    env.episodeScore = 0
    while True:
        selectedAction, actionProb = agent.work(state, type_="selectActionMax")
        state, reward, done, _ = env.step([selectedAction])
        env.episodeScore += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            state = env.reset()
