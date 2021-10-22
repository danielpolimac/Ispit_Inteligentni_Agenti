## Training a trading agent using reinforcement learning

![alt text](https://www.fairobserver.com/wp-content/uploads/2020/03/markets-2.jpg)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries listed in requirements.txt.

```bash
pip install -r requirements.txt

```

## Background

The stock market is characterized by rapid change, many interference factors and insufficient periodic data. Stock trading is a game process under
incomplete information, and the single-objective supervised learning model
is difficult to deal with such serialization decision problems. Reinforcement
learning is one of the effective ways to solve such problems.

Trading environment for this project was simulated using AnyTrading library. It has implementation for two markets: FOREX and Stock. AnyTrading aims to provide some Gym environments to improve and facilitate the procedure of developing and testing RL-based algorithms in this area. This purpose is obtained by implementing three Gym environments: TradingEnv, ForexEnv, and StocksEnv, of which the latest was used for the purpose of this experiment.

The goal of the agent is to maximize the value of his portfolio at the end of a trading period and outperform a benchmark using a Deep Reinforcement Learning model in a single stock trading. The agent observes a current state the environment shows and chooses a trading action from the action space. The actions available to the agent are buy and sell. By performing these actions agent is holding either the long or the short position. In long position agent wants to buy shares when prices are low and profit by sticking with them while their value is going up, and in short position agent wants to sell shares with high value and use this value to buy shares at a lower value, keeping the difference as profit.
For more detailed explanation of financial positions please check this Wikipedia article:
https://en.wikipedia.org/wiki/Position_(finance)

The Agent (A2C) is based on actor critic architecture using Kronecker-factored Trust Region, which was developed by researchers at the University of Toronto and New York University, and implemented in Open Baselines library by the OpenAI team. The algorithm combines a few key ideas:
An updating scheme that operates on fixed-length segments of experience and uses these segments to compute estimators of the returns and advantage function, architectures that share layers between the policy and value function, and asynchronous updates.

## Dataset
The data used in this experiment comes from Huge Stock Market Dataset, hosted on Kaggle as a courtesy of Boris Marjanovic. The dataset consists of individual stocks and ETFs. For the purpose of this experiment I've chosen to work with stock of Tesla, an electric automotive company, covering the span of their business activity between 2010 and 2017 Dataset has the following columns: Date,Open,High,Low,Close,Volume. Complete dataset can be found following this link:
https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

## Usage

```python
# Import libraries
import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import quantstats as qs

# Load the dataset and set the date column as index
df = pd.read_csv('tesla.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Instantiate the training environment
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,730), window_size=5)
env = DummyVecEnv([env_maker])

# Instantiate a learning agent and define a learning policy:
model = A2C('MlpLstmPolicy', env, verbose=1)

# Start the training
model.learn(total_timesteps=1000000)

# Evaluate the trained model against new data
env = gym.make('stocks-v0', df=df, frame_bound=(730, 910), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

# Plot the results
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
```
![alt-text](https://github.com/AminHP/gym-anytrading/raw/master/docs/output_14_1.png)


## Analyze the agent's trading decisions using Quantstats

QuantStats is a Python library that performs portfolio profiling, allowing quants and portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics. For the purpose of examining  agent's trading decision I used QuantStats' reports module in order to generate metrics reports, batch plotting, and creating tear sheets that can be saved as an HTML file.

```python

import quantstats as qs
qs.extend_pandas()
window_size = 5
start_index = 730
end_index = 910

net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='a2c_quantstats.html')
```
The report generated should look something like this:
![alt text](https://github.com/ranaroussi/quantstats/raw/main/docs/report.jpg?raw=true)


## References
Papers:

V. Mnih et al, Asynchronous Methods for Deep Reinforcement Learning, ICML, 2016. 
https://arxiv.org/abs/1602.01783

Tools:

https://stable-baselines.readthedocs.io/en/master

https://gym.openai.com/

https://github.com/AminHP/gym-anytrading

https://github.com/ranaroussi/quantstats


## License
[MIT](https://choosealicense.com/licenses/mit/)
