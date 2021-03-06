import matplotlib.pyplot as plt
import time
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
while(1):
	fp=open("Episode-score.txt", "r")
	lines = fp.read().splitlines()
	scores = list(map(float, lines))
	episode = list(range(1, 1+len(scores)))
	scores = scores[:11000]
	episode = episode[:11000]
	print(scores)
	print(episode)
	
	plt.figure()
	plt.title("Episode scores over episode")
	plt.plot(episode, scores, label='Raw data')
	SMA = moving_average(scores, 100)
	plt.plot(SMA, label='SMA100')
	plt.xlabel("episode")
	plt.ylabel("episode score")
	#plt.show()
	plt.legend()
	plt.savefig('trend.png')
	time.sleep(2)
