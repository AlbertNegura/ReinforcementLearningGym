# ReinforcementLearningGym
ACML Assignment 3

In order to run mountaincar.py, you need a compatible Pyhton version (3.4 or later) and these libraries installed:
 - Numpy
 - Matplotlib
 - OpenAI Gym


Example command prompt execution: python mountaincar.py -a 0.5 -g 0.7 --episodes 100000 -b 50 -s 3000 --agent SARSA

Parameters' explanation:
-a (float) : alpha value, the learning rate from each experience (Default = 0.1)
-g (float) : gamma value, the discount factor for predicted future gains (Default = 0.9)
-e, --episodes (int) : number of epochs the simulation should run (Default = 10000)
-r, --render (bool) : toggles the GUI window to render mountain car using the OpenAI Gym environment (Default = False)
-b, --bins (int) : number of bins used in discretisizing both position and velocity space (Default = 70)
-s, --steps (int) : number of time steps each epoch should take (Default = 2000)
--agent (string) : selection of algorithm to train the agent (Default = QLearning) --- (options: QLearning, SARSA, Both)
BEWARE: Selecting 'Both' for --agent parameter runs QLearning first, furthermore, when running agents of both algorithms, you can't set separate parameters for each.
