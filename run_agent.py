# Ucitavanje biblioteka
import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Ucitavanje i priprema podataka
df = pd.read_csv('tesla.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Instanciranje okruženja za trening
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,730), window_size=5)
env = DummyVecEnv([env_maker])

# Instanciranje i trening modela
# Opis A2C agenta https://stable-baselines.readthedocs.io/en/master/modules/a2c.html
model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

# Evaluacija obučenog modela
env = gym.make('stocks-v0', df=df, frame_bound=(730, 910), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

# Graficki prikaz rezultata
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

# Analiza finansijskih rezultata modela uz pomoc quantstats biblioteke
window_size = 5
start_index = 730
end_index = 910

import quantstats as qs
qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='a2c_quantstats.html')