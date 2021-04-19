import matplotlib.pyplot as plt
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def Plot(text_path="./exports/", dest_path="./exports/"):
    fp=open(text_path+"Episode-score.txt", "r")
    lines = fp.read().splitlines()
    scores = list(map(float, lines))
    episode = list(range(1, 1+len(scores)))
    scores = scores[0:5000]
    episode = episode[0:5000]
    plt.figure()
    plt.title("Episode scores over episode")
    plt.plot(episode, scores, label='Raw data')
    SMA = moving_average(scores, 10)
    plt.plot(SMA, label='SMA10')
    plt.xlabel("episode")
    plt.ylabel("episode score")
    
    plt.legend()
    plt.savefig(dest_path+'trend.png')
    print("Last SMA500:",np.mean(scores[-150:]))

def Plot2(text_path="./exports/", dest_path="./exports/"):
    fp=open(text_path+"Episode-score2.txt", "r")
    lines = fp.read().splitlines()
    scores = list(map(float, lines))
    episode = list(range(1, 1+len(scores)))
    scores = scores[0:300*5000]
    episode = episode[0:300*5000]
    plt.figure()
    plt.title("Step punishment over episode")
    plt.plot(episode, scores, label='Raw data')
    # SMA = moving_average(scores, 10)
    # plt.plot(SMA, label='SMA10')
    plt.xlabel("step (1 epoch = 300 step)")
    plt.ylabel("step punishment")
    
    plt.legend()
    plt.savefig(dest_path+'trend2.png')
    print("Last SMA500:",np.mean(scores[-150:]))

if __name__ == '__main__':
    Plot()
    Plot2()
