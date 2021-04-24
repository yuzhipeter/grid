import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

world = 'Easy'
pd_list = []
label_list = []
for lr in [0.01, 0.1, 0.5]:
    for epsilon in [0.3, 0.5, 0.7]:
        filename = world + ' Q-Learning L{:0.2f} E{:0.1f}.csv'.format(lr, epsilon)
        pd_temp = pd.read_csv(filename)
        #print(pd_temp.head(5))
        pd_list.append(pd_temp)
        label_list.append('Learning Rate={:.2f}, Epsilon={}'.format(lr, epsilon))

pd_p = pd.read_csv(world + ' Policy.csv')
pd_v = pd.read_csv(world + ' Value.csv')


# Q-learning plots
# convergence
#line_styles = [':',':',':','--','--','--','-','-','-']
#colors = ['blue','lime','red','blue','lime','red','blue','lime','red']
line_styles = [':','--','-',':','--','-',':','--','-']
colors = ['blue','blue','blue','lime','lime','lime','red','red','red']
plt.figure()
for pd, label, line_style, color in zip(pd_list, label_list, line_styles, colors):
    x = pd['iter'].values
    y = pd['convergence'].values
    plt.plot(x, y, label=label, linestyle=line_style, color=color, linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Q-value difference')
    plt.title('Q-learning: Convergence check')
plt.savefig(world + '_QLearning_convergence.png', format='png', dpi=300)
plt.show()

# reward
line_styles = [':','--','-',':','--','-',':','--','-']
colors = ['blue','blue','blue','lime','lime','lime','red','red','red']
fig, ax1 = plt.subplots()
for pd, label, line_style, color in zip(pd_list, label_list, line_styles, colors):
    x = pd['iter'].values
    y = pd['reward'].values
    ax1.plot(x[:400], y[:400], label=label, linestyle=line_style, color=color, lw=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.legend(loc='lower center', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x[400:], y[400:], label=label, linestyle=line_style, color=color, lw=0.3)
    ax2.set_ylim([0,100])
    ax2.set_ylabel('Reward', color='red')
    ax2.tick_params('y', colors='red')
    plt.title('Q-learning: Reward history')
plt.savefig(world + '_QLearning_reward.png', format='png', dpi=300)
plt.show()


# reward
line_styles = [':','--','-',':','--','-',':','--','-']
colors = ['blue','blue','blue','lime','lime','lime','red','red','red']
fig, ax1 = plt.subplots()
for pd, label, line_style, color in zip(pd_list, label_list, line_styles, colors):
    x = pd['iter'].values
    y = pd['steps'].values
    ax1.plot(x[:400], y[:400], label=label, linestyle=line_style, color=color, lw=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Steps')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.legend(loc='upper center', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x[400:], y[400:], label=label, linestyle=line_style, color=color, lw=0.3)
    ax2.set_ylim([0,100])
    ax2.set_ylabel('Steps', color='red')
    ax2.tick_params('y', colors='red')
    plt.title('Q-learning: Steps')
plt.savefig(world + '_QLearning_Steps.png', format='png', dpi=300)
plt.show()
       

# time
line_styles = [':','--','-',':','--','-',':','--','-']
colors = ['blue','blue','blue','lime','lime','lime','red','red','red']
plt.figure()
for pd, label, line_style, color in zip(pd_list, label_list, line_styles, colors):
    x = pd['iter'].values
    y = pd['time'].values
    y = np.cumsum(y)
    plt.plot(x, y, label=label, linestyle=line_style, color=color, linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Compuational Cost (ms)')
    plt.title('Q-learning: Compuational Cost')
plt.savefig(world + '_QLearning_time.png', format='png', dpi=300)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Value & policy iterations

# convergence
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['convergence'].values
x2 = pd_v['iter'].values
y2 = pd_v['convergence'].values
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Policy & Value Iteration: Convergence check')
plt.savefig(world + '_PV_convergence.png', format='png', dpi=300)
plt.show()

# reward
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['reward'].values
x2 = pd_v['iter'].values
y2 = pd_v['reward'].values
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Reward')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.set_ylim([0,100])
ax2.set_ylabel('Reward', color='red')
ax2.tick_params('y', colors='red')
plt.title('Policy & Value Iteration: Reward history')
plt.savefig(world + '_PV_reward.png', format='png', dpi=300)
plt.show()

# steps
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['steps'].values
x2 = pd_v['iter'].values
y2 = pd_v['steps'].values
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Step')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.set_ylim([0,100])
ax2.set_ylabel('Step', color='red')
ax2.tick_params('y', colors='red')
plt.title('Policy & Value Iteration: Step')
plt.savefig(world + '_PV_steps.png', format='png', dpi=300)
plt.show()

# time
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['time'].values
y1 = np.cumsum(y1)
x2 = pd_v['iter'].values
y2 = pd_v['time'].values
y2 = np.cumsum(y2)
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Computational Cost (ms)')
plt.title('Policy & Value Iteration: Computational Cost')
plt.savefig(world + '_PV_time.png', format='png', dpi=300)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Comparison

# convergence
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['convergence'].values
x2 = pd_v['iter'].values
y2 = pd_v['convergence'].values
x3 = pd_list[5]['iter'].values[:200]
y3 = pd_list[5]['convergence'].values[:200]
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
plt.plot(x3, y3, label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Comparison: Convergence check')
plt.savefig(world + '_Comparison_convergence.png', format='png', dpi=300)
plt.show()

# reward
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['reward'].values
x2 = pd_v['iter'].values
y2 = pd_v['reward'].values
x3 = pd_list[5]['iter'].values[:200]
y3 = pd_list[5]['reward'].values[:200]
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
ax1.plot(x3[:40], y3[:40], label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Reward')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.plot(x3[60:], y3[60:], label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
ax2.set_ylim([0,100])
ax2.set_ylabel('Reward', color='red')
ax2.tick_params('y', colors='red')
plt.title('Comparison: Reward history')
plt.savefig(world + '_Comparison_reward.png', format='png', dpi=300)
plt.show()

# steps
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['steps'].values
x2 = pd_v['iter'].values
y2 = pd_v['steps'].values
x3 = pd_list[5]['iter'].values[:200]
y3 = pd_list[5]['steps'].values[:200]
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
ax1.plot(x3[:40], y3[:40], label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Step')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.plot(x3[60:], y3[60:], label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
ax2.set_ylim([0,100])
ax2.set_ylabel('Step', color='red')
ax2.tick_params('y', colors='red')
plt.title('Comparison: Reward history')
plt.savefig(world + '_Comparison_steps.png', format='png', dpi=300)
plt.show()

# time
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['time'].values
y1 = np.cumsum(y1)
x2 = pd_v['iter'].values
y2 = pd_v['time'].values
y2 = np.cumsum(y2)
x3 = pd_list[5]['iter'].values[:100]
y3 = pd_list[5]['time'].values[:100]
y3 = np.cumsum(y3)
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
plt.plot(x3, y3, label='Q-learning, lr=0.1, epsilon=0.7', color='red', linewidth=1.)
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Computational Cost (ms)')
plt.title('Comparison: Computational Cost')
plt.savefig(world + '_Comparison_time.png', format='png', dpi=300)
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
world = 'Hard'
pd_list_hard = []
label_list_hard = []
for lr in [0.01, 0.1, 0.5]:
    for epsilon in [0.3, 0.5, 0.7]:
        filename = world + ' Q-Learning L{:0.2f} E{:0.1f}.csv'.format(lr, epsilon)
        pd_temp = pd.read_csv(filename)
        #print(pd_temp.head(5))
        pd_list_hard.append(pd_temp)
        label_list_hard.append('Learning Rate={:.2f}, Epsilon={}'.format(lr, epsilon))

pd_p_hard = pd.read_csv(world + ' Policy.csv')
pd_v_hard = pd.read_csv(world + ' Value.csv')


# Q-learning plots
# convergence
line_styles = [':','--','-',':','--','-']
colors = ['blue','blue','blue','red','red','red']
pds = pd_list[2:9:3] + pd_list_hard[2:9:3]
labels = label_list[2:9:3] + label_list_hard[2:9:3]
plt.figure()
for pd, label, line_style, color in zip(pds, labels, line_styles, colors):
    x = pd['iter'].values
    y = pd['convergence'].values
    plt.plot(x, y, label=label, linestyle=line_style, color=color, linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Q-value difference')
    plt.title('Q-learning: Convergence check')
plt.savefig(world + '_QLearning_convergence.png', format='png', dpi=300)
plt.show()

# reward
line_styles = [':','--','-',':','--','-']
colors = ['blue','blue','blue','red','red','red']
pds = pd_list[2:9:3] + pd_list_hard[2:9:3]
labels = label_list[2:9:3] + label_list_hard[2:9:3]
fig, ax1 = plt.subplots()
for pd, label, line_style, color in zip(pds, labels, line_styles, colors):
    x = pd['iter'].values
    y = pd['reward'].values
    ax1.plot(x[:400], y[:400], label=label, linestyle=line_style, color=color, lw=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.legend(loc='lower center')

    ax2 = ax1.twinx()
    ax2.plot(x[400:], y[400:], label=label, linestyle=line_style, color=color, lw=0.3)
    ax2.set_ylim([0,100])
    ax2.set_ylabel('Reward', color='red')
    ax2.tick_params('y', colors='red')
    plt.title('Q-learning: Reward history')
plt.savefig(world + '_QLearning_reward.png', format='png', dpi=300)
plt.show()


# steps
line_styles = [':','--','-',':','--','-']
colors = ['blue','blue','blue','red','red','red']
pds = pd_list[2:9:3] + pd_list_hard[2:9:3]
labels = label_list[2:9:3] + label_list_hard[2:9:3]
fig, ax1 = plt.subplots()
for pd, label, line_style, color in zip(pds, labels, line_styles, colors):
    x = pd['iter'].values
    y = pd['steps'].values
    ax1.plot(x[:400], y[:400], label=label, linestyle=line_style, color=color, lw=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Steps')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.legend(loc='upper center', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x[400:], y[400:], label=label, linestyle=line_style, color=color, lw=0.3)
    ax2.set_ylim([0,100])
    ax2.set_ylabel('Steps', color='red')
    ax2.tick_params('y', colors='red')
    plt.title('Q-learning: Steps')
plt.savefig(world + '_QLearning_Steps.png', format='png', dpi=300)
plt.show()
       

# time
line_styles = [':','--','-',':','--','-']
colors = ['blue','blue','blue','red','red','red']
pds = pd_list[2:9:3] + pd_list_hard[2:9:3]
labels = label_list[2:9:3] + label_list_hard[2:9:3]
plt.figure()
for pd, label, line_style, color in zip(pds, labels, line_styles, colors):
    x = pd['iter'].values
    y = pd['time'].values
    y = np.cumsum(y)
    plt.plot(x, y, label=label, linestyle=line_style, color=color, linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Compuational Cost (ms)')
    plt.title('Q-learning: Compuational Cost')
plt.savefig(world + '_QLearning_time.png', format='png', dpi=300)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Value & policy iterations

# convergence
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['convergence'].values
x2 = pd_v['iter'].values
y2 = pd_v['convergence'].values
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
x3 = pd_p_hard['iter'].values
y3 = pd_p_hard['convergence'].values
x4 = pd_v_hard['iter'].values
y4 = pd_v_hard['convergence'].values
plt.plot(x3, y3, label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
plt.plot(x4, y4, label='Value Iteration', color='lime', linewidth=1., linestyle='--')
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Policy & Value Iteration: Convergence check')
plt.savefig(world + '_PV_convergence.png', format='png', dpi=300)
plt.show()

# reward
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['reward'].values
x2 = pd_v['iter'].values
y2 = pd_v['reward'].values
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
x3 = pd_p_hard['iter'].values
y3 = pd_p_hard['reward'].values
x4 = pd_v_hard['iter'].values
y4 = pd_v_hard['reward'].values
ax1.plot(x3[:40], y3[:40], label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
ax1.plot(x4[:40], y4[:40], label='Value Iteration', color='lime', linewidth=1., linestyle='--')
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Reward')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.plot(x3[60:], y3[60:], label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
ax2.plot(x4[60:], y4[60:], label='Value Iteration', color='lime', linewidth=1., linestyle='--')
ax2.set_ylim([0,100])
ax2.set_ylabel('Reward', color='red')
ax2.tick_params('y', colors='red')
plt.title('Policy & Value Iteration: Reward history')
plt.savefig(world + '_PV_reward.png', format='png', dpi=300)
plt.show()

# steps
fig, ax1 = plt.subplots()
x1 = pd_p['iter'].values
y1 = pd_p['steps'].values
x2 = pd_v['iter'].values
y2 = pd_v['steps'].values
ax1.plot(x1[:40], y1[:40], label='Policy Iteration', color='blue', linewidth=1.)
ax1.plot(x2[:40], y2[:40], label='Value Iteration', color='lime', linewidth=1.)
x3 = pd_p_hard['iter'].values
y3 = pd_p_hard['steps'].values
x4 = pd_v_hard['iter'].values
y4 = pd_v_hard['steps'].values
ax1.plot(x3[:40], y3[:40], label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
ax1.plot(x4[:40], y4[:40], label='Value Iteration', color='lime', linewidth=1., linestyle='--')
plt.legend(loc='best')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Step')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc='center')

ax2 = ax1.twinx()
ax2.plot(x1[60:], y1[60:], label='Policy Iteration', color='blue', linewidth=1.)
ax2.plot(x2[60:], y2[60:], label='Value Iteration', color='lime', linewidth=1.)
ax2.plot(x3[60:], y3[60:], label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
ax2.plot(x4[60:], y4[60:], label='Value Iteration', color='lime', linewidth=1., linestyle='--')
ax2.set_ylim([0,100])
ax2.set_ylabel('Step', color='red')
ax2.tick_params('y', colors='red')
plt.title('Policy & Value Iteration: Step')
plt.savefig(world + '_PV_step.png', format='png', dpi=300)
plt.show()

# time
plt.figure()
x1 = pd_p['iter'].values
y1 = pd_p['time'].values
y1 = np.cumsum(y1)
x2 = pd_v['iter'].values
y2 = pd_v['time'].values
y2 = np.cumsum(y2)
plt.plot(x1, y1, label='Policy Iteration', color='blue', linewidth=1.)
plt.plot(x2, y2, label='Value Iteration', color='lime', linewidth=1.)
x3 = pd_p_hard['iter'].values
y3 = pd_p_hard['time'].values
y3 = np.cumsum(y3)
x4 = pd_v_hard['iter'].values
y4 = pd_v_hard['time'].values
y4 = np.cumsum(y4)
plt.plot(x3, y3, label='Policy Iteration', color='blue', linewidth=1., linestyle='--')
plt.plot(x4, y4, label='Value Iteration', color='lime', linewidth=1., linestyle='--')
plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Computational Cost (ms)')
plt.title('Policy & Value Iteration: Computational Cost')
plt.savefig(world + '_PV_time.png', format='png', dpi=300)
plt.show()
