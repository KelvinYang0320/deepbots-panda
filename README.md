# Franka Emika Panda with Deepbots and Reinforcement Learning
> [deepbots](https://github.com/aidudezzz/deepbots)\
> Deepbots is a simple framework which is used as "middleware" between the free and open-source Cyberbotics' Webots robot simulator and Reinforcement Learning algorithms. When it comes to Reinforcement Learning the OpenAI gym environment has been established as the most used interface between the actual application and the RL algorithm. Deepbots is a framework which follows the OpenAI gym environment interface logic in order to be used by Webots applications.
## Installation
1. [Install Webots](https://www.cyberbotics.com/)
2. Install Python versions 3.7 or 3.8
    * Follow the Using Python guide provided by Webots
3. Install deepbots through pip running the following command:\
<code>pip install deepbots</code>
4. Install PyTorch via pip
5. Clone this repository by running the following command:\
<code>git clone https://github.com/KelvinYang0320/deepbots-panda.git</code>

## How it works

### Overview: emitter - receiver scheme
![image](https://github.com/KelvinYang0320/deepbots-panda/blob/master/img/deepbots_overview.png)
### Class Diagram with PPO agent for Cartpole
![image](https://github.com/KelvinYang0320/deepbots-panda/blob/master/img/classDiagram.png)
### Observation
To do better than take random actions at each step, we ant to know what our actions are doing to the environment.
The environment’s <code>supervisor.step</code> function returns exactly what we need. In fact, step returns four values. These are:
* <code>observation</code> (object): an environment-specific object representing your observation of the environment.
* <code>reward</code> (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
* <code>done</code> (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
* <code>info</code> (dict): diagnostic information useful for debugging.

### Action
Decide on what action should be taken by <code>agent.work()</code>.

## Current Achievements and work
### Achievements: Reach a Target via PPOAgent
![image](https://github.com/KelvinYang0320/deepbots-panda/blob/master/img/demo.gif)
![image](https://github.com/KelvinYang0320/deepbots-panda/blob/master/img/trend.png)
### Reinforcement Learning Algorithn
1. PPO
> https://openai.com/blog/openai-baselines-ppo/
2. DQN
> [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518 (7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
3. DoubleDQN
> [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)

TBC...

reference: https://github.com/Curt-Park/rainbow-is-all-you-need