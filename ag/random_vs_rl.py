import mxnet as mx
import autogluon as ag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print(f"mxnet version: {mx.__version__}")
print(f"autogluon version: {ag.__version__}")


def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

config_size = 100
num_trials = 280

x, y = np.linspace(0, config_size - 1, config_size), np.linspace(0, config_size - 1, config_size)
X, Y = np.meshgrid(x, y)

Z = np.zeros(X.shape)
ps = [(20, 70, 35, 40, 1),
      (80, 40, 20, 20, 0.7)]
for p in ps:
    Z += gaussian(X, Y, *p)


print(f"Shapes: {X.shape}, {Y.shape}, {Z.shape}")
print(f"max(Z) = {Z.max()}")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
ax.set_zlim(0,np.max(Z)+2)
# plt.show()
plt.savefig('./gmm.png')
plt.close()


@ag.args(
    x=ag.space.Categorical(*list(range(config_size))),
    y=ag.space.Categorical(*list(range(config_size))),
)
def rl_simulation(args, reporter):
    x, y = args.x, args.y
    reporter(accuracy=Z[y][x])


random_scheduler = ag.scheduler.FIFOScheduler(rl_simulation,
                                              resource={'num_cpus': 1, 'num_gpus': 0},
                                              search_options={"random_seed": 412},
                                              num_trials=num_trials,
                                              reward_attr='accuracy')
random_scheduler.run()
random_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(random_scheduler.get_best_config(), random_scheduler.get_best_reward()))


import tempfile
tmp_chkp_dir = tempfile.mkdtemp()

rl_scheduler = ag.scheduler.RLScheduler(rl_simulation,
                                        resource={'num_cpus': 1, 'num_gpus': 0},
                                        num_trials=num_trials,
                                        reward_attr='accuracy',
                                        search_options={"random_seed": 42},
                                        controller_batch_size=4,
                                        controller_lr=5e-3,
                                        checkpoint=f'{tmp_chkp_dir}/checkerpoint.ag')
rl_scheduler.run()
rl_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(rl_scheduler.get_best_config(), rl_scheduler.get_best_reward()))


results_rl = np.array([v[0]['accuracy'] for v in rl_scheduler.training_history.values()])
results_random = np.array([v[0]['accuracy'] for v in random_scheduler.training_history.values()])

results1 = results_random.reshape(-1, 10).mean(axis=1)
results2 = results_rl.reshape(-1, 10).mean(axis=1)

print(results1)
print(results2)

plt.plot(range(len(results1)), results1, range(len(results2)), results2)
plt.savefig('./results.png')





