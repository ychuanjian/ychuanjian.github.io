import matplotlib.pyplot as plt
import numpy as np

max_steps = 10000
max_runs = 100
# step_start = 100000
mode_nc = 'non-constant'
mode_con = 'constant'
# mode_grad = 'gradient'
parameter_setting = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3, 4]


class Policy(object):
    def __init__(self, mode=mode_con, q0=0., epsilon=0.1, action_space=10, alpha=0.1, baseline=True, part=1.):
        self.action_space = action_space
        self.q0 = q0 * part
        self.q_estimate = np.array([q0] * action_space)
        self.mode = mode
        self.baseline = baseline
        self.n = 0
        self.c = 2 * part
        self.pi = np.zeros(action_space)
        self.reward_bar = 0
        self.h_action = np.zeros(action_space)
        self.alpha = alpha * part
        self.epsilon = epsilon * part
        self.num_selected = np.array([0] * action_space)
        self.description = ''
        # self.description += mode + ', '
        self.description += 'epsilon = ' + str(self.epsilon)
        if mode == mode_nc:
            self.description += ', alpha = 1/n'
        elif mode == mode_con:
            self.description += ', alpha = ' + str(self.alpha)
        else:
            print('Policy mode wrong!!')
            quit()
        if epsilon == -2:   # gradient
            if baseline:
                self.description += ', with baseline'
            else:
                self.description += ', without baseline'

        self.pi[np.random.choice(action_space)] = 1.
        # self.average_reward = np.mean(self.q_estimate)

    def freshen(self):
        self.q_estimate = np.array([self.q0] * self.action_space)
        self.num_selected = np.array([0] * self.action_space)
        self.h_action[:] = 0.

    def choose_action(self):
        a = 0
        self.n += 1
        if self.epsilon >= 0:   # epsilon greedy
            if np.random.random() < self.epsilon:  # non-greedy
                a = np.random.randint(0, self.action_space)
            else:
                a = np.argmax(self.q_estimate)
        elif self.epsilon == -1:  # upper confidence bound : c = 2
            if not all(self.num_selected):
                # print(np.where(self.num_selected == 0))
                a = np.random.choice(np.where(self.num_selected == 0)[0])
                # print(a)
            else:
                a = np.argmax(self.q_estimate + self.c * np.square(np.log(self.n) / self.num_selected))
        elif self.epsilon == -2:    # gradient
            a = np.random.choice(self.action_space, p=self.pi)
        self.num_selected[a] += 1
        return a

    def update_q(self, action, reward):
        q_old = self.q_estimate[action]
        if self.mode == mode_nc:
            self.alpha = 1. / self.num_selected[action]
        if self.epsilon == -2:
            h_a_old = self.h_action.copy()
            self.h_action = h_a_old - self.alpha * (reward - self.reward_bar) * self.pi
            self.h_action[action] = h_a_old[action] + self.alpha * (reward - self.reward_bar) * (1 - self.pi[action])
            self.pi = np.exp(self.h_action)/sum(np.exp(self.h_action))
            if self.baseline:
                reward_bar_old = self.reward_bar
                self.reward_bar = reward_bar_old + 1. / self.n * (reward - reward_bar_old)
        else:
            self.q_estimate[action] = q_old + self.alpha * (reward - q_old)
        # self.average_reward = np.mean(self.q_estimate)
        # print(self.average_reward, self.q_estimate)


class VirtualEnv(object):
    def __init__(self, action_space=10, mean=0, var=1):
        self.mean = mean
        self.variance = var
        self.action_space = action_space
        self.values = np.random.normal(mean, var, action_space)
        self.optimal_action = np.argmax(self.values)
        # print(self.values, self.optimal_action)

    def freshen(self):
        self.values = np.random.normal(self.mean, self.variance, self.action_space)
        self.optimal_action = np.argmax(self.values)

    def fluctuate(self, mean=0, var=0.01):
        disturb = np.random.normal(mean, var, self.action_space)
        self.values += disturb
        self.optimal_action = np.argmax(self.values)

    def feedback(self, action):
        return np.random.normal(self.values[action], 1)


average_reward = {}
optimal_action_rates = {}


def main():
    # policies = [Policy(epsilon=-2, alpha=0.1, baseline=True), Policy(epsilon=-2, alpha=0.4, baseline=True),
    # Policy(epsilon=-2, alpha=0.1, baseline=False), Policy(epsilon=-2, alpha=0.4, baseline=False)]
    # policies = [Policy(mode='non-constant'), Policy(mode='constant')]
    # policies = [Policy(mode='constant', q0=5., epsilon=0.), Policy(q0=0., epsilon=0.1, mode='constant')]
    policies = []

    # num_optimal = np.array([[0] * max_steps] * len(policies))
    reward_sum = np.array([[0.] * len(parameter_setting)] * 4)
    reward_sum_per_run = np.array([[[0.] * len(parameter_setting)] * 4] * max_runs)
    env = VirtualEnv(mean=0)
    for j in range(1, max_runs + 1):
        if j % 10 == 0:
            print('run : ', j)
        env.freshen()
        for param in parameter_setting:
            policies.clear()
            policies.append(Policy(part=param))  # epsilon-greedy
            policies.append(Policy(epsilon=-2, part=param))  # gradient
            policies.append(Policy(epsilon=-1, part=param))  # ucb
            policies.append(Policy(q0=5., part=param))  # optimistic initial
            index_y = parameter_setting.index(param)
            for pi in policies:
                # pi.freshen()
                index_x = policies.index(pi)
                # print(index_y)
                for i in range(max_steps):
                    action = pi.choose_action()
                    reward = env.feedback(action)
                    env.fluctuate()
                    pi.update_q(action, reward)
                    # reward_sum[index_x, index_y] += reward
                for i in range(max_steps):
                    action = pi.choose_action()
                    reward = env.feedback(action)
                    env.fluctuate()
                    pi.update_q(action, reward)
                    reward_sum[index_x, index_y] += reward
            reward_sum_per_run[j - 1, :, index_y] = reward_sum[:, index_y]
    average_reward['epsilon-greedy'] = np.mean(reward_sum_per_run[:, 0, :], axis=0)
    average_reward['gradient bandit'] = np.mean(reward_sum_per_run[:, 1, :], axis=0)
    average_reward['upper bound confidence'] = np.mean(reward_sum_per_run[:, 2, :], axis=0)
    average_reward['optimistic initial'] = np.mean(reward_sum_per_run[:, 3, :], axis=0)
    """for index in range(len(policies)):
        pi = policies[index]
        average_reward[pi.description] = 1. * reward_sum[index] / max_runs
        optimal_action_rates[pi.description] = 1. * num_optimal[index] / max_runs"""


if __name__ == '__main__':
    main()
    # print(average_reward)
    plt.subplot(211)
    for p in average_reward:
        plt.plot(average_reward[p], label=p)
    plt.xlabel('epsilon,alpha,c,Q0')
    plt.xticks(range(0, len(parameter_setting)), labels=[str(p) for p in parameter_setting])
    # plt.axis([1, max_steps, 0, 1.6])
    plt.ylabel('Average reward over first ' + str(max_steps) + ' steps')
    plt.legend()
    """plt.subplot(212)
    for p in optimal_action_rates:
        plt.plot(optimal_action_rates[p], label=p)
    plt.xlabel('Steps')
    # plt.axis([1, max_steps, 0, 1.5])
    plt.ylabel('% Optimal action')
    plt.legend()"""
    plt.show()
    # print(optimal_action_rates['epsilon = 0.0, alpha = 0.1'][:30])

