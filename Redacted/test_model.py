import gym
import numpy as np
import pylab
import test_brain_fakeenv
import test_LAWC
import time
from threading import Thread


brain = test_brain_fakeenv.Brain()
# brain = test_LAWC.Brain()
training_steps = 1000

# do this for ever
def run_batch():
    score = 0
    cnt = 0
    done = False
    state = brain.env.reset()
    # while not done:
    #     cnt+=1
    #     training_thread = Thread(target=brain.update_model)
    #     training_thread.start()
    #     brain.initialize_model()
    #     while training_thread.is_alive() and not done:
    #         brain.env.render()
    #         action, state, done, reward = brain.predict(state)
    #         score += reward
    #     training_thread.join()
    # brain.update_model()

    while not done:
        cnt+=1
        brain.initialize_model()
        while not done:
            brain.env.render()
            action, state, done, reward = brain.predict(state)
            score += reward
            brain.update_model()


        # while not done:
        #     brain.env.render()
        #     action, state, done, reward = brain.predict(state)
        #     score += reward

    return score

scores_, episodes_, average_ = [], [], []
max_average = 0
pylab.figure(figsize=(18, 9))
pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
def PlotModel(score, episode):
    scores_.append(score)
    episodes_.append(episode)
    average_.append(sum(scores_[-50:]) / len(scores_[-50:]))
    pylab.plot(episodes_, scores_, 'b')
    pylab.plot(episodes_, average_, 'r')
    print("Average: ", average_[-1])
    pylab.ylabel('Score', fontsize=18)
    pylab.xlabel('Steps', fontsize=18)
    try:
        pylab.grid(True)
        pylab.savefig("test_AC.png")
    except OSError:
        pass

start = time.time()
for e in range(10000):
    print("new batch")
    score = run_batch()
    PlotModel(score, e)

end = time.time()
print(end-start)

brain.env.close()





