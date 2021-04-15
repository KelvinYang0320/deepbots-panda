# Franka Emika Panda with Deepbots and Reinforcement Learning
> [deepbots](https://github.com/aidudezzz/deepbots)\
> Deepbots is a simple framework which is used as "middleware" between the free and open-source Cyberbotics' Webots robot simulator and Reinforcement Learning algorithms. When it comes to Reinforcement Learning the OpenAI gym environment has been established as the most used interface between the actual application and the RL algorithm. Deepbots is a framework which follows the OpenAI gym environment interface logic in order to be used by Webots applications.

## Installation
1. [Install Webots](https://www.cyberbotics.com/)
2. Install Python versions 3.7 or 3.8
    * Follow the Using Python guide provided by Webots
3. Install deepbots through pip running the following command:\
<code>pip install [deepbots 0.1.2](https://pypi.org/project/deepbots/)</code>
4. Install PyTorch via pip
5. Clone this repository by running the following command:\
<code>git clone https://github.com/KelvinYang0320/deepbots-panda.git</code>

## How it works

### Overview: emitter - receiver scheme
![image](https://github.com/KelvinYang0320/deepbots-panda/blob/Panda-deepbots-0.1.2/img/deepbots_overview.png)

### Action
Decide on what action should be taken by <code>agent.work()</code> for the positions of 7 motors.

## Current Achievements and work
### Achievements: Reach a Target via PPOAgent
|Trained Agent Showcase|Reward Per Episode Plot|
|---|---|
|![image](https://github.com/KelvinYang0320/deepbots-panda/blob/Panda-deepbots-0.1.2/img/demo.gif)|![image](https://github.com/KelvinYang0320/deepbots-panda/blob/Panda-deepbots-0.1.2/img/trend.png)|
