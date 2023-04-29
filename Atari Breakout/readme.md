# My Attempt At Attari Breakout
In the field of artificial intelligence, one of the most exciting and challenging areas of research is reinforcement learning. This is a type of machine learning where an agent learns how to take actions in an environment in order to maximize a reward. One popular approach to reinforcement learning is deep Q-networks (DQNs), which are neural networks that learn to approximate the optimal action-value function.

I've attempted to use a DQN network (deep Q-network) to solve an Atari Breakout game, but after running about 2M frames without solving the same, I've assuming something is off with either the algoritem or the network. I've used a ConvLSTM2D network that would allow the network to capture both the direction of the movment by inserting 4 frames as a series, and 2 layers of convelotion as in the Deepmind paper [1]. 
 
I later learned that DeepMind ren around 45 million frames to solve Atari Breakout, which they were able to achieve by running their algorithm on 16 threads in around 10 to 12 hours. Since I was was able to run about 1 million frames within 2 hours on a single thread it would have taken me around 45 hours to verify if my algorithm actually work.

Despite the failure, I learned a lot from this experience. I realized that instead of pursuing Breakout, which I don't think is feasible for a DQN network with a single thread to solve in reasonable time. Instead, I aim try to Atari Pong with a Rainbow Network. The Rainbow Network is an extension of the DQN architecture that incorporates several techniques such as dueling networks, prioritized experience replay, and distributional reinforcement learning [3].

![image](https://user-images.githubusercontent.com/29729128/235302481-315f9624-f6a9-45d8-bb8e-398b55b6b1ee.png)

Figure 1: Comparing Deepmind Atari Breakout & Pong learning time (16 threads)[2]


![image](https://user-images.githubusercontent.com/29729128/235301809-0e869d3d-7a83-4464-9708-4918f8a6fa08.png) [3]

Figure 2: Comparing Deepmind Atari Breakout & Pong learning curve (the first x-tick is at 50M frames) DQN - Gray, Rainbow - white

According to Deepmind chart Pong is solvable within 7M frames, with a steeper learning curve that should show some results within 1M frames, which should take my computer running on 1 thread 2 hours. I also assumed that I could tune the hyperparameters more specifically to Pong, such as setting a faster learning rate and learning based on black and white frames instead of grayscale. These should theoretically allow me to complete Pong with even faster time.

I leave this code here a reference for future Pong work.
* Altough the experience is realy, this content was assisted with a Generative model (chartGPT).

1. [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [Asynchronous Methods for Deep Reinforcement Learning
](https://arxiv.org/pdf/1602.01783.pdf)
3. [Rainbow: Combining Improvements in Deep Reinforcement Learning
](https://arxiv.org/pdf/1710.02298.pdf)
