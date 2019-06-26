---
layout: post
title: CHAPTER 5. Monte Carlo Methods
date: 2019-06-13 
tags: 强化学习
mathjax: true
---

本章开始学习第一个估计值函数与找到最优策略的方法，蒙特卡洛  
蒙特卡洛法不要求完全的环境动态，而是使用**经验**(*experience*)---与环境的真实或模拟交互得到的 states、actions and rewards 的样本  
真实经验效果惊人，它不需要环境动态的任何先验知识  
模拟经验也很强大，虽然依然需要一个环境模型，但是这个模型只需要提供状态转移与奖励的样本，而不是如 DP 一般，需要所有状态转移可能的完整概率分布  
令人惊讶的是，很多时候，一个所求概率分布的经验样本生成器是很容易得到的，而关于它的具体分布的获取却难之又难  

蒙特卡洛法基于平均样本回报来解决 RL 问题  
为了得到具体的回报，将蒙特卡洛法只用在 episodic tasks 中，即无论如何选择 action，任务总会在某些点结束  
只有在一个周期结束后，值函数和策略才会改变，蒙特卡洛法是 episode-by-episode 提升的，而非 step-by-step(online)  
蒙特卡洛广义上是指，涉及到重要的随机分量操作的估计方法  
在这里，用它来特指使用 完整回报的均值 的方法  

蒙特卡洛法 对每个 state-action 对 的回报进行采样平均，类似于在第2章中，对每个 action 的 reward 进行采样平均  
主要区别在于，这里有多个状态，每个动作都相当于多个 bandit problem，并且这些不同的 bandit problem 是有内在关联的  
也就是，选择一个动作后，它的回报与同一周期内后续 状态中采取的动作是有依赖关系的  
因为所有动作选择都会经历学习，对于比较早的状态而言，问题就变成了 nonstationary  

解决 nonstationarity，采用了 GPI，在第4章，我们利用 MDP 的环境知识来计算值函数  
在这里，我们会从 MDP 的回报采样中来学习 值函数  
值函数和策略达到最优的方式在本质上是一样的  
在 DP 中，使用了 policy evaluation 和 policy improvement ，蒙特卡洛其实也是一样的，只是环境动态在这里变成了经验的采样  

### 5.1 Monte Carlo Prediction ###
也就是 DP 中的 policy evaluation，从一个给定的 policy 中，学到它对应的值函数  
状态的值其实就是，该状态所能获得的 回报 的期望  
从经验中估计值，最简单的办法就是在多次经历某一状态后，将它最终获得的回报均值化  
当得到的回报样本足够多，均值便会趋于回报了  
所有蒙特卡洛方法的背后都是这个简单的思想  

假设：对一个策略 $\pi$ 估计其在状态 $s$ 下的值函数 $v_\pi(s)$ ，并给出一系列 episodes 的集合  
在一个周期中，每次 $s$ 出现都称作一次 *visit to s* ，在一个周期中，同一个状态可能有多个 visit，其中第一次则叫做 *first-visit to s*  
*First-visit MC method*: 只用每个周期中的第一次 visit to s 来作为样本，均值化之后估计其值  
*Every-visit MC method*: 一个周期中的每一次 visit to s 都作为样本  
其中 first-visit MC method 用得更广，也是本章的重点  
而 every-visit MC method 在 函数逼近中的扩展较多  
首次访问蒙特卡洛法的伪代码如下，每次访问蒙特卡洛法的代码与其相同，只是不再检查 s 是否首次出现  

![first_visit_MC_prediction](/assets/images/RL-Introduction/Chapter5/first_visit_MC_prediction.png)

两种 MC 方法均会随 visits 次数 使估计逐渐逼近值函数  
该例中，每个 回报 都是关于 拥有有限方差的 $v_\pi(s)$ 的独立同分布的估计量  
根据大数定理，序列的均值会收敛于它们的期望值  
每个均值本身都是一个 标准差落在 $1/\sqrt{n}$ 内的无偏估计量，n 是回报的数量  
Every-visit MC 没有这么直接，但它的估计值也会以二次方式收敛于 $v_\pi(s)$ (Singh and Sutton, 1996)  

### Example 5.1： Blackjack ###
黑杰克也就是 21 点的规则就不多介绍了  
设每局游戏都是一次 episode ，获胜则 reward 即 return 为 1，否则为 0  

在该例中，环境动态其实是已知的，但是它非常的复杂，难以计算且很容易出错  
使用 DP 方法可行但太过麻烦，而使用 MC 则显得更加合适  
因为构造一个黑杰克的环境模型其实并不复杂，虽然它对应的环境动态很复杂  
在很多时候，即便环境动态是可计算的，也不会使用 DP 去解决问题，因为环境动态一般而言都很难算  

MC 中，每个状态的估计值是相互独立的  
对一个状态的估计不是建立在任何其它状态的估计之上的  

特别的，对单个状态的值估计所需的计算量消耗，是与状态数量无关的  
当我们只需要计算某个状态或某些状态的值时，MC 是一个不错的选择  
我们可以只生成那些我们感兴趣的状态开始的 episode ，然后用来估计值，而忽略掉其它用不到的状态  

### 5.2 Monte Carlo Estimation of Action Values ###
如果模型不可得，那么估计 动作值函数 会比 状态值函数要 有用些  
如果有模型，那么只要有状态值，就能够很有效地得出一个策略，只要简单地进行单步搜索即可，就像 DP 中做的那样  
如果没有模型，那么只有 状态值是不够的，因为单步搜索时没有转移后的状态，无法估计出动作的值  
而想要得到一个策略，必须要准确地估计出每个动作的值  

估计 $q_\pi(s,a)$ 的方法其实与估计 $v_\pi(s)$ 的方法是一样的，只是把 visit to s 变成了 visit to s-a  
估计动作值函数有一个问题：很多 state-action 对可能一次都没有被访问  
比如当 $\pi$ 是一个确定型的策略时，每个状态只能观察到一个动作  
由于缺少足够的数据，以至于并不能提升策略  
一个通用思路是**持续探索** (*maintaining Exploration*)   
我们可以让起点在所有的 state-action 对 中随机选择，这样在无数次尝试中总是能够遍历所有的 state-action 对  
这种方法被称为**探索起点** (*exporing Starts*)  

探索起点并不总是可靠的，特别是在直接从实际与环境交互中学习的情况下，这种情况下的起点条件往往不起到什么作用  
更通用的办法应该是使得所有的 state-action 对都会被考虑到，这需要一个随机策略，在每个状态都有非零的概率选中任意的动作  

### 5.3 Monte Carlo Control ###
接下来考虑如何让 MC 估计 用于控制中，以得到一个近似的最优策略  
方法与 DP 中的类似，也引入了 GPI ，在 evaluation 与 improvement 的交替中提升策略  
区别仅在于，MC 中总是保持着 近似的 值函数 与 策略，可以看做是数值解而非解析解  
另有一点就是这里用的是 动作值 而非 状态值，因为没有环境动态  

这里做了两个假设，一是使用了探索起点，另一个是无限个 episode  
想要得到一个能够实用的算法，必须解除两个假设；本节接下来会解除后者，前者在后面解决  

其实在 DP 中就已经遇到过这个问题，当 策略或值函数 只在无穷远处才会收敛时，有两个办法来解决：  
* 一是置信度的方法，首先测量或假设估计中误差的大小和概率的界限，然后通过足够多的实验来保证这个界限是足够小的  
这一方法可以用来保证估计能够收敛到什么程度，但它依然需要大量的 episodes 来保证可信  
* 二是类似于 DP 中的 value iteration，不指望 policy evaluation 能够有多可靠， 在有限的 steps 后就直接从 evaluation 转到 improvement  

下图是使用探索起点的 MC 算法，在任意的 state-action 对出现后直接更新其策略，这其实与 value iteration 类似  

![Monte_Carlo_ES](/assets/images/RL-Introduction/Chapter5/Monte_Carlo_ES.png)

### 5.4 Monte Carlo Control without Exploring Starts ###
解除探索起点的假设，主要思想是保证 agent 能够持续不断地选中任意的动作  
有两种办法来保证这一点：  
*on-policy*：估计、改进用于做决策的 policy  
*off-policy*：估计、改进的 policy 不同于生成数据的那个 policy  
上一节的算法用的是 on-policy  
这一节将设计一种不需要探索起点假设的 on-policy 算法  
下一节来讨论 off-policy  

on-policy 中，policy 是软的 (soft)  
先保证对任意的 $s\in \mathcal S,a\in \mathcal A$ ,有 $\pi(a|s)>0$ ，然后再逐渐变成一个确定性的最优策略  
第二章中的许多鼓励探索的方法都能用于此，这里我们用 $\epsilon$-greedy   
对于所有 nongreedy action，选择概率为 $\frac{\epsilon}{|\mathcal A(s)|}$， 而greedy action 的选中概率为 $1-\epsilon +\frac{\epsilon}{|\mathcal A(s)|}$  
而 $\epsilon$-soft 则是对于所有的 states 和 actions ，有 $\pi(a|s)\geq \frac{\epsilon}{|\mathcal A(s)|}$， 其中 $\epsilon>0$  
显然，任意的 $\epsilon$-greedy 策略至少会等于 $\epsilon$-soft 策略，更大的可能是优于

在不用探索起点的情况下，无法简单地用贪婪策略来改进旧策略，因为这会阻止对那些 nongreedy actions 的探索  
幸运的是，GPI 并不要求策略是一个完全的贪婪策略，只要它在向贪婪策略靠近即可  
在以下算法中，我们将它往一个 $\epsilon$-greedy 策略靠近  
对于任意的 $\epsilon$-soft policy $\pi$, 关于 $q_\pi$ 的任意 $\epsilon$-greedy policy 都能保证优于或等于 $\pi$   

![on_policy_first_visit_MC_control_for_epsilon_soft_policies](/assets/images/RL-Introduction/Chapter5/on_policy_first_visit_MC_control_for_epsilon_soft_policies.png)

该算法可以保证每次 improvement 后的策略会优于或等于旧策略，并且仅在新旧两个策略都是最优的 $\epsilon$-soft 策略时才收敛  
考虑一个新的环境，新环境与之前有相同的动作和状态集，并按如下所述行动  
如果在 s 选择动作 a，那么新环境有 $1-\epsilon$ 的可能与旧环境采取相同的行为，有 $\epsilon$ 的可能等概率选择其它的动作，然后按照旧环境继续行动  
在这个新环境下，最好的选择就是采用与旧环境相同的 $\epsilon$-soft 策略  
设 $\tilde v_\ast$ 和 $\tilde q_\ast$ 表示新环境中的最优值函数  
那么对于一个策略 $\pi$ ,当且仅当 $v_\pi=\tilde{v}_\ast$ 时为最优 $\epsilon$-soft 策略  

以上说明了 $\epsilon$-soft 策略的 policy iteration 算法  
在 $\epsilon$-soft 策略中用 greedy 思想来改进  

### 5.5 Off-policy Prediction via Importance Sampling ###
所有的学习控制方法都面临一个窘境：追寻着最优行为的值函数，却不得不选择非最优的行为来进行探索    
要如何在探索策略下学得最优策略呢？  
之前所讲的 on-policy 方法做了一种妥协：不去追求最优策略，而是一个近似最优的探索型策略  
更直接的办法是使用两种策略：一种学习称为最优策略；另一种则更具备探索性以生成行为  
所要学习的叫做**目标策略**(*target policy*)，而用于生成行为的叫做**行动策略**(*behavior policy*)  

On-policy方法：一般更简单且通常是有限考虑的  
Off-policy方法：需要额外的概念说明，且数据来源于其它策略，因此会波动较大且收敛更慢，但它更强大、更通用  
可以把 on-policy 看做是 off-policy 的一种特例，其 target policies 和 behavior policies 是相同的策略  
Off-policy 有更多的实际应用，其学习的数据可以来自那些传统的控制器产生或者由人类专家直接给出  
Off-policy 也被看做是学习环境动态的多步预测模型的关键  

本节中通过一个 *prediction* problem 来学习 off-policy 方法，其 target 、behavior policies 都是固定的  
样例中，我们需要估计 $v_\pi$ 或 $q_\pi$ ，而我们所有的 episodes 数据确实来源于另一个策略 $b\neq \pi$ ，该例中，$\pi$ 是 目标策略， $b$ 是行为策略，这两个策略都是给定的  
为了使用来自 $b$ 的 episodes 来估计 $\pi$ 的值，需要使得 $\pi$ 中选取的每个动作都在 $b$ 中被选中过  
即需要 $\pi(a\mid s)>0\quad implies\quad b(a\mid s)>0$   
这叫做覆盖假定 assumption of *coverage*   
由覆盖看出：在与 $\pi$ 不相同的状态中， $b$ 必须是随机策略(**为什么呢？**)；另一方面，目标策略 $\pi$ 可以是确定的，而这一点是控制应用所感兴趣的。  
在控制中，目标策略是关于对 动作值函数的现有估计的一个确定性贪婪策略；在行为策略保持随机并更具备探索性(如 $\epsilon$-greedy policy)的情况下，目标策略会变成一个确定性的最优策略  
而本节中，我们考虑的预测问题中的策略 $\pi$ 是给定的  

**重要性采样**(*importance sampling*) 是一种通用技术，用于估计一个分布的期望值，而它的采样来自另外一个分布  
几乎所有的 off-policy 都利用了重要性采样  
根据目标与行为策略下，轨迹出现的相关概率，为 returns 加上一个权重，叫做**重要性采样率**(*importance-sampling ratio*)  
给定初始状态 $S_t$ ，在任意策略 $\pi$ 下，后续的状态动作轨迹的概率：  

$$\begin{align}
Pr\{A_t&,S_{t+1},A_{t+1},\dots,S_T\mid S_t,A_{t:T-1}\sim \pi\} \\
&=\pi(A_t\mid S_t)p(S_{t+1}\mid S_t,A_t)\pi(A_{t+1}\mid S_{t+1})\cdots p(S_T\mid S_{T-1},A_{T-1}) \\
&= \prod_{k=t}^{T-1}\pi(A_k\mid S_k)p(S_{k+1}\mid S_k,A_k)
\end{align}$$   

其中 $p$ 为状态转移概率，那么重要性采样率为：  

$$\rho_{t:T-1}\doteq \frac{\prod_{k=t}^{T-1}\pi(A_k\mid S_k)p(S_{k+1}\mid S_k,A_k)}{\prod_{k=t}^{T-1}b(A_k\mid S_k)p(S_{k+1}\mid S_k,A_k)}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}$$

虽然轨迹概率与未知的 MDP 的转移概率相关，但是被消掉了；最终的重要性采样率仅与两个策略以及序列相关，而与 MDP 无关了  

我们想要估计的是在目标策略下的 期望回报值，而我们有的却是在 行为策略下的 回报 $G_t$，由此得到的是错误的回报  $\mathbb E[G_t\mid S_t]=v_b(S_t)$ ，这不能用于做平均来得到 $v_\pi$  
这是为何使用 重要性采样 的原因，重要性采样率 $\rho_{t:T-1}$ 改变了回报，从而得到正确的期望值：  

$$\mathbb E[\rho_{t:T-1}G_t\mid S_t]=v_\pi(S_t)$$

现在已经可以给出一个蒙特卡洛算法，通过观测策略 $b$ 给出的一系列 episodes，平均它们的回报值，来估计 $v_\pi(s)$  
这里可以很方便地跨越 episode 来给 time steps 进行编号：  
如果第一个 episode 终结于 time 100， 那么下一个 episode 就开始于 time $t=100$  
这使我们能够用 time-step 来指向 特定 episode 中的 特定 step  
另外，可以定义 $\mathcal J(s)$ 为 访问过状态 s 的所有 time steps 的集合，用于 every-visit 方法  
而对于 first-visit 方法，$\mathcal J(s)$ 仅包含每个 episode 中第一次访问 s 的 time step  
令 $T(t)$ 表示在 t 之后第一个终结状态的 time step，$G_t$ 表示从 t 到 $T(t)$ 的回报  
那么 $\{G_t\}_{t\in \mathcal J(s)}$ 为属于状态 s 的回报，$\{\rho_{t:T(t)-1}\}_{t\in \mathcal J(s)}$ 表示对应的重要性采样率  
为了估计 $v_\pi(s)$ ，简单地把重要性采样率乘上回报，平均后得到结果：  

$$
V(s)\doteq \frac{\sum_{t\in \mathcal J(s)}\rho_{t:T(t)-1}G_t}{|\mathcal J(s)|}
$$

重要性采样用在这样简单的平均方式中，叫做 **原始重要性采样**(*ordinary importance sampling*)  
一个重要的变种叫做 **加权重要性采样**(*weighted importance sampling*)，使用了加权平均：  

$$
V(s)\doteq \frac{\sum_{t\in \mathcal J(s)}\rho_{t:T(t)-1}G_t}{\sum_{t\in \mathcal J(s)}\rho_{t:T(t)-1}}
$$

分母为零时定义为零  

考虑在得到一个 回报 后的估计：加权-平均估计中，分子分母中的 $\rho_{t:T(t)-1}$ 消掉了，那么平均回报等于该回报值  
考虑到该回报是仅有的，这个估计值很合理，但它却是 $v_b(s)$ 而非 $v_\pi(s)$ 的期望，在这种统计意义上它有偏差  
相比而言，简单平均的期望始终是 $v_\pi(s)$ 的，但却可能变得很极端  
假设 重要性采样率为 10， 表明在目标策略中，观测到该轨迹的可能性是在行为策略中观测到该轨迹概率的 10 倍；如此一来，原始重要性采样的估计值将会是观测到的回报的 10 倍，即使这条轨迹在 目标策略中非常典型，这个值与观测到的回报相差也太过悬殊了  

两种重要性采样之间的差别可以用它们的偏差和方差来表示  
原始重要性采样是无偏的，但方差一般是无界的，因为 ratios 可以是无界的
而加权重要性采样是有偏差的(偏差会渐趋于 0 )，在任意单个回报上的方差都是1；实际上，假设回报是有界的，那么加权重要性采样估计值的方差会趋于 0 ，即便 ratios 是无限大 (Precup,Sutton and Dasgupta 2001)  
在实际应用中，加权法与原始法相比，方差会小非常多；加权法更受到人们的青睐  
原始法没有被放弃是因为它可以很方便地扩展到逼近方法中，这在本书的第二部分  

有时可以通过简单的计算，证明 importance-sampling-scaled returns 是无限的；  
使用随机变量的方差公式：  

$$Var[X]\doteq\mathbb E\big[(X-\bar X)^2\big]=\mathbb E[X^2-2X\bar X+\bar X^2]=\mathbb E[X^2]-\bar X^2.$$  

那么，如果均值是有限的，当且仅当 随即变量 $X$ 的平方的期望值是无限时， $X$ 的方差会是无限的；那么我们只要证明 $\mathbb E[X^2]$ 是无限的，即：  

$$\mathbb E_b\Big[\Big(\prod_{t=0}^{T-1}\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}G_0\Big)^2\Big]$$

在例 5.5 中就可以用该式来证明 importance-sampling-scaled returns 是无限的  

### 5.6 Incremental Implementation ###
基于 episode-by-episode ，可以在 平均 returns 时使用 Incremental Implementation  
对于 off-policy MC，需要分别考虑 原始重要性采样 与 加权重要性采样  
原始重要性采样是简单地将 returns 以 ratio $\rho_{t:T(t)-1}$ 缩放后，再做均值操作；可以将第二章中用的 incremental implementation 用在其上，只要把 rewards 替换为 缩放后的 returns 即可  
对于 加权重要性采样，我们需要构建一个稍有不同的 incremental implementation  
假设有一序列的回报值 $G_1,G_2,\dots,G_{n-1}$ ,起始于同一个状态，并各自对应一个随机的权重 $W_i=\rho_{t:T(t)-1}$ ，希望构建一个估计为：  

$$V_n\doteq \frac{\sum_{k=1}^{n-1}W_kG_k}{\sum_{k=1}^{n-1}W_k}$$  

并在收到一个新的 $G_n$ 后即时更新，为了跟踪 $V_n$ 的轨迹，必须保存每个状态的前 n 个回报对应权重的累计和 $C_n$ ，更新规则为：  

$$\begin{align}
&V_{n+1}\doteq V_n+\frac{W_n}{C_n}[G_n-V_n],\quad n\geq 1,\\
&C_{n+1}\doteq C_n+W_{n+1}
\end{align}$$  

其中 $C_0\doteq 0$ (而 $V_1$ 为任意值)  
一个完整的 MC policy evaluation 的 episode-by-episode incremental 算法如下，算法虽然显示的是用于 off-policy，使用 weighted importance sampling； 但同样可以用于 on-policy，只要令 $\pi=b，W 恒为 1$ 即可  
虽然所有的 动作 均取自于 策略 $b$ ，但其最终的近似 $Q$ 会收敛于 $q_\pi$ (所有经历过的 状态-动作 组合)  

![off_policy_MC_prediction_incremental_policy_evaluation](/assets/images/RL-Introduction/Chapter5/off_policy_MC_prediction_incremental_policy_evaluation.png)

### 5.7 Off-policy Monte Carlo Control ###
Off-policy 的一个优势就是 target policy 可能是确定性的(贪婪)，而 behavior policy 却能够采样所有的可能动作  
Off-policy MC control methods 会用到前两节所讲的技巧，它们都使用 behavior policy 学习来改善 target policy  
这些技巧要求 behavior policy 在那些所有可能被 target policy 选中的动作上拥有非零的选中概率 (coverage)  
为了探索所有的可能性，要求 behavior policy 是 soft-policy (所有状态下选择所有的动作的概率都是非零)  

下图是 off-policy MC control method 的算法伪代码  
基于 GPI、weighted importance sampling，用于估计 $\pi_\ast$ 和 $q_\ast$ ，目标策略 $\pi\approx \pi_\ast$ 是一个关于 $q_\ast$ 的估计 $Q$ 的贪婪策略  
behavior policy $b$ 可以是任意策略，但应当能够保证 $\pi$ 可以收敛到最优策略，对所有的 状态-动作 组合能够得到无穷多的回报采样，这可以用 $\epsilon$-soft 策略来保证  
策略 $\pi$ 能够在所有经历的状态上收敛，即便策略 $b$ 在 episode 之间乃至其中发生了变化  

![off_policy_MC_control_for_estimating_pi](/assets/images/RL-Introduction/Chapter5/off_policy_MC_control_for_estimating_pi.png)

一个潜在的问题：该方法仅在 episodes 的尾部进行学习，此时该 episode 剩余的所有 actions 均为 greedy  
如果 nongreedy actions 频繁出现，那么学习速度就会变慢，特别是在较长的 episodes 的前期出现的状态  
这一点有时会大大降低算法的学习速度，如果这个问题很严重，最重要的方式就是将其适当地调整为 temporal-difference learning ，另外就是，当 $\gamma<1$ 时，它也能起到巨大的帮助作用  

### 5.8 \*Discounting-aware Importance Sampling ###

### 5.9 \*Per-dicision Importance Sampling ###

### 5.10 Summary ###
本章介绍的 蒙特卡洛法 通过得自 sample episodes 的经历来学习 值函数 与 最优策略  
该方法相对于 DP 有至少三个优势：  
1. 可以再没有环境动态模型的情况下，通过与环境的交互直接学习到最优行为  
2. 可以用在 模拟仿真 或 采样模型 中；在 DP 方法中，有些时候模型的状态转移概率非常复杂而难以构造，这时用 MC 方法会惊人得简单  
3. 在 MC 中，可以更有效地关注于状态的子集，有时候精确估计某些状态集的代价会很大，而 MC 方法可以绕开这个缺点  

在本书的后续章节，会讲到 MC 方法的第四个优势：对 不满足马尔科夫属性 情况下，收到的损失更小  
这时因为 MC 不会在后续状态的 value estimates 上 来更新 value estimates；也就是说，MC 不用 bootstrap  

设计 MC control methods 时应遵循 GPI； GPI 表示了 policy evaluation 和 policy improvement 的交替运行  
MC 有一个异种 policy evaluation process； 不是用模型计算每个状态的 值，而是直接从 某状态开始，采样得到它们的回报并均值化，因为一个状态的 值 就是期望回报，该均值很好地逼近了 value  
在 control methods 中，我们更关注 逼近 动作值函数，因为它能在不需要环境动态模型的情况下 改进 策略  
MC 方法在 episode-by-episode 的基础上混合了 policy evaluation 和 policy improvement 的 steps，并且能使用 incrementally implementation  

保持有效探索是一个关键  
只选择那些当前估计最好的动作是不够的，对于那些不曾获得过采样回报的 动作，不做探索就永远无法让策略变得更好  
一个简单的办法就是通过让 episode 开始于 随机的 state-action 对，以此来覆盖所有的可能性  
在某些模拟的 episode 应用中，exploring starts 方法有时会起作用，但在实际的学习中并不适用  
在 on-policy 中，agent 会始终保持探索，并试图在探索中找到 最优策略  
而在 off-policy 中，agent 依然会探索，但学习到的 确定性最优策略 也许是与 所遵循策略没有联系的  

Off-policy prediction 是指通过由一个 behavior policy 产生的数据 来学习一个 target policy 的 值函数，而 behavior policy 与 target policy 是不同的两个策略  
这种学习方法是基于某些 重要性采样 方法的，通过用 ratio-两种策略能观测到的动作的概率 来对 回报 进行 加权操作，来将 behavior policy 的期望 转变成 target policy 的期望  
原始重要性采样 对 加权回报 使用简单的均值化操作，会产生一个 无偏估计，但是有更大的、乃至于无限的 方差  
而 加权重要性采样 使用了 加权平均， 它产生的 方差 总是有限的，因此更加实用  
尽管 off-policy 在概念上非常简单，但是 off-policy MC 中无论是 prediction 还是 control 都是一个悬而未决的难题  

MC 与 DP 的区别主要有两点：  
1. MC 使用了 采样经验，由此可以在没有模型的情况下进行学习  
2. MC 不用 bootstrap，也就是它不会基于其它的 值估计 来更新某些 值估计  

这两点联系不深，甚至可以被分开看待，下一章将介绍 只含第一点而不含第二点区别的 方法，也就是既使用 采样经验，也使用 bootstrap 的方法  
