import matplotlib.pyplot as plt
import time
while(1):
	fp=open("Episode-score.txt", "r")
	lines = fp.read().splitlines()
	scores = list(map(float, lines))
	episode = list(range(1, 1+len(scores)))
	print(scores)
	print(episode)
	plt.figure()
	plt.plot(episode, scores)
	plt.xlabel("Episode",fontsize=13,fontweight='bold')
	plt.ylabel("Score",fontsize=13,fontweight='bold')
	#plt.show()
	plt.savefig('trend.png')
	time.sleep(2)
