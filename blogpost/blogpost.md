# Choosing an Experience-Replay Method for Deep Q-learning

Reinforcement learning has gained a lot of popularity in recent years due some spectacular successes due to the successful integration of Deep Learning into Reinforcement Learning. This breakthrough substantially improved the capacity of such models to learn complex policies for sequential decision-making problems, famously demonstrated by the deep Q network (DQN), exhibiting expert play on Atari 2600 video game, or by AlphaGo, defeating the best human player at the game of Go. In this blog-post we will discuss a key ingredient of DQN that contributed to the success of these networks called "experience replay". 


### First of all, what is DQN?

DQN, is exactly what the name suggests; the deep-learning neural network used as a non-linear approximator of the Q-function. Just to remind ourselves, here, the "Q" in Q-learning stands for quality, representing how useful an action is, given the current state, with respect to gaining future reward. Therefore, we want to learn a mapping between state-action pairs, and their respective q-values. Now our implementation of DQN will be a two-layer deep neural network





The parameters $\textbf{w}$ of the network are then updated using the following update rule 




Now our implementation of DQN will be a two-layer deep neural network, used as a non-linear approximation of the state-action value function (Q function). Therefore, it maps state-action pairs to their respective q-values. The parameters $\textbf{w}$ of the network are then updated using the following update rule 
 
 
### So, Why Do We Need Replay Methods?

Experience replay is a mechanism enabling agents to memorize and reuse past experiences multiple times. 
This has been inspired from "memory replay", a biological process that occurs in the brain during sleep, important for memory processing and consolidation.
In practice, experience replay is implemented by storing the training data in a memory buffer and subsequently, replay/process the "memories" offline. We can interpret this as our agent remembering/daydreaming past experiences.
This has been empirically shown to improve learning efficiency, and stabilize the training by breaking the correlations between experiences. 

<br>
**TODO:**format picture

 ![](figs/dreaming.jpg)
caption: Robots dream too!

<br>

Typically, off-policy algorithms assume a uniform sampling strategy, replaying the different past experiences with equal frequency.
Intuitively, our agent can learn better and more efficiently by prioritizing the processing of task-relevant experiences over that of redundant/irrelevant experiences. 
However, being relevant can be subjective, and therefore we can assume that different sorting criteria can be important in different situations; that is, not every situation could benefit optimally using the same kind of replay.
We will discuss a few cases were you could choose for a particular replay method.

### How can we analyze the effect of different Experience Replay Methods?


### Let's experiment!



## Why Some Replay Methods Work Better for Some Environments:

**TODO:** *discuss per environment or per method why the winner won where it did... (thereby merge methods- and results sections for more of a blog-post feeling)*

