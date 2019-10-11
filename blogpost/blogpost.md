# Choosing a Memory-Replay Method for Deep Q-learning

A recent breakthrough in AI has been the successful integration of RL with DL, which substantially improved the capacity of such models to learn complex policies for sequential decision-making problems. 
This has been famously demonstrated by the deep Q network (DQN), exhibiting expert play on Atari 2600 video game, or by AlphaGo, defeating the best human player at the game of Go. 
 

### What is deep Q-learning?

The former algorithm, deep Q-learning, is exactly what the name suggests; the Deep-learning variant of the famous Q-learning algorithm. 
This algorithm tries to optimize the expected value of state-action pairs given its current estimate of the best next state after. It does so model-free, meaning it does not depend on a interpretation of the environment.
 
### So, Why Do We Need Replay Methods?

A key ingredient in these networks is "experience replay", which enables agents to memorize and reuse past experiences multiple times. 
This has been inspired from "memory replay", a biological process that occurs in the brain during sleep, important for memory processing and consolidation.
In practice, experience replay is implemented by storing a subset of the training data in a memory buffer and subsequently, replay/process the "memories" offline. 
This has proven to improve sample efficiency, and stabilizing the training by breaking the correlations between experiences. 
Besides, it helps a network "remember" what happened some time ago.

Typically, off-policy algorithms assume a uniform sampling strategy, replaying the different past experiences with equal frequency.
However, not every situation requires the same kind of replay.
Intuitively, our agent can learn better and more efficiently by prioritizing the processing of task-relevant experiences over that of redundant/irrelevant experiences. 
We will discuss a few cases were you could choose for a particular replay method.


## Why Some Replay Methods Work Better for Some Environments:

**TODO:** *discuss per environment why the winner won... (merge methods and results sections for more of a blog-post feeling)*

