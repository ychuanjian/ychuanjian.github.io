---
layout: post
title: CHAPTER 1. INTRODUCTION
date: 2019-05-17 16:53:30
tags: 强化学习
mathjax: true
---

RL是通过与环境进行交互行为，并通过计算来学习的一种方法；相比较其它机器学习方法，更关注于交互，直指实验目标。  
**RL要做的是**：学会做什么：即如何从当前situation来得到action，来最大化reward  
learner需要通过不断的尝试来发现哪个action能够产生较大的reward  
较复杂的情况：actions不仅影响reward，还会影响**下一个situation**，并且由此影响到后续的rewards  
RL的两个显著特征：*trial-and-error*(摸石头过河)、*delayed reward* (延后的反馈)  
Reinforcement Learning 既能表示增强学习的问题，也能表示解决它的方法，还能表示研究这类问题及其解决办法的领域  
需要注重区分这类表达方式，避免造成混淆，尤其要分清：增强学习问题和解决方法的区别  
可以用动态系统-dynamic system 理论 (数学概念) 中的观点来描述增强学习   
具体来说就是：the optimal control of incompletely-known Markov decision processes.	(Chapter 3)

**一个学习代理必须具备三个条件：**
* 能够感知周围环境的状态 state
* 能够做出影响 state 的行为 action
* 一个与 state 相关的目标 goal

Markov decision processes 包含了这三个方面：sensation、action、goal

**RL与其它机器学习方法**

**监督学习：**  
* 由具备知识的外部监督者 supervisor 提供标记过的数据集，用于学习
* 目标：在训练集以外的数据上，能够做出准确的判断
* 监督学习不适用于交互

**非监督学习：**  
* 寻找未标记数据中的隐含结构信息

**强化学习**  
* 获得最大的奖励 reward

**RL的挑战：**  
*trade-off between explration and exploitation*  
这个问题一直没有得到解决  
RL还有一个特征，就是关注于在代理与环境的交互中直接解决目标问题  
其它机器学习常常关注大环境中的子问题，尽管这同样出了许多喜人的成果，但这总归是一个很大的限制  
RL则相反，代理总是会关注一个明确的目标，能够感知环境并做出影响环境的行为，环境通常具有很大的不确定性  
当RL涉及规划时，它必须解决规划与实际行为对环境造成的影响之间产生的相互作用，即环境变化对原定规划的影响，以及规划中对环境变化的判断，同时环境本身模型的改进与调整也是一个问题  
当RL涉及监督学习时，需要分离出一个子问题，这个子问题应当在代理中扮演一个明确的角色，哪怕其在完整代理中无法填充细节  
RL的代理并不是单纯指向机器人一类的完整结构，也可以指向一个更大系统中的一部分，此时它与整个系统中的其它部分产生直接交互，与该系统所处环境产生间接交互  
人们必须超越显而易见的代理人及其环境的例子来理解强化学习框架的普遍性  

**RL的四个要素**：策略 *a policy* 、奖励信号 *a reward signal* 、价值函数 *a value function* 、环境模型 *a model of the environment*  
* policy：定义了在什么场景 state 下应当做出什么行为 action ，有时会是一个搜索的过程。一般而言，策略可能是随机的
* reward signal：定义了RL的学习目标，决定了行为的好坏，是策略更新的基础。一般而言，reward signal 是state 和 action 的随机函数
* value function：RL的长期目标，在某一个 state +action 下，其产生的 reward 可能是低的，但其后续造成的影响能带来更高的 value ，这是RL所需要的  
    actions 的最终目标总是获取更高的values，但是 reward 在做出action 后可以由环境直接得出，而values 却需要估算且未必准确，有效估算values 的方法是RL的核心问题  
    value 的另一个理解：为了保证获得更多的 total reward ，所追求的一个长期目标  
    这里将 total reward 认为是RL的终极目标，而某些时候选择一些低 reward 的行为带来了更高的 value， 而高的 value 能够保证获得更高的total reward  
    比如将 reward 视作人得到的愉悦，value 视作人生命的长度，那么高 value 可以保证得到更多的愉悦，但终极目标还是 total reward  
    即在某些极高的 reward 面前，可以放弃 value ，就像人为了某些东西放弃生命一样，因为这可以带来极高的愉悦
* model of the environment：在model-based RL 中会使用一个环境模型来预测环境的变化，这个模型不是真正的环境；而不使用环境模型，直接走一步看一步的RL方法称为 model-free 方法

### RL的局限与范围

RL强烈依赖于state ，state 可以理解为代理所在环境能够给出的任何信息，它告诉代理当前的环境是如何的
(state 本身非常重要，但本书更关注于如何根据state 给出action)  
大部分RL方法围绕着estimating value function 来构造，然而estimating value function 并不是必须的  
如遗传算法、遗传编程、模拟退火及一些其它优化方法就不去估计价值函数。  
它们提供了多个统计策略，每个策略在一段长时间内各自与独立的环境实体交互，那些得到最高reward 的策略及其变种进入下一轮迭代  
这种算法称为进化方法，当策略空间足够小，或者搜索时间足够长，进化算法能够取得有效的成果。  
另外，进化算法在代理无法完全感知环境状态的问题中具有优势。  
我们的重点是强化学习方法，这些方法在与环境相互作用时学习，而进化方法则不然  
能够利用个体交互行为细节的方法在许多场合会比进化算法更有效。  
进化算法忽略了许多RL问题的有用的结构：  
* 忽略了所搜索的策略是一个从states 到 actions 的函数
* 忽略了个体在其生命周期中传递的具体states 或 选择的具体 actions

### Example：Tic-Tac-Toe

假设对手有可能失误，要求找到对手的失误，使获胜的机会最大  
虽然这个问题很简单，但是使用传统的方法并不容易得到满意的结果  
minimax：会避免失败，但也会错失获胜机会  
顺序决策问题的经典优化：如动态规划，可以为任何对手计算最优解，但需要输入该对手的完整规范，包括对手在每个棋盘状态下进行每次移动的概率。  
大部分情况下，环境的完整信息无法提供。关于在这个问题上可以做的最好的事情是首先学习对手行为的模型，达到某种程度的置信度，然后应用动态规划来计算给定近似对手模型的最优解。  
进化算法在该问题上会直接搜索可能的策略空间，找到一个获胜概率高的策略。这里的策略就是告诉player 对应每个游戏中的 state ，下一步应该写在哪里  
对每个策略，都能够在与对手下棋的过程中获得一个对获胜概率的估计，这个估计能够在之后指导选择哪些策略  
一个典型的进化算法就像在策略空间中爬山，在努力获得提升的过程中产生并估计策略，或者说，一个遗传类算法能够用于保持和估计一系列策略  
使用value function：  
首先为游戏中的所有可能的state 设置一组数表，每个数表示从对应状态获取胜利的可能性的最新估计，把这些估计座位state 的 value，整个数表就是学习到的value function  
value 值高表明该状态获胜的机会大，己方已经三连的状态value 为1，对方已经达成三连子的状态value 则为0，其它状态的初始value 设为0.5  
与对手进行对战，在落子时，检查每一个可能的落点对应的 states 并查表得到对应 value  
大多数时候遵循贪婪法则，即总是选择胜率最高的下法，这就是 exploit ；偶尔地，在不是最高胜率的下法中随机选择一种，以获取新知识。这是 explore  
在对战落子时，每次会更新state 对应的value ，尽量做到精确估计获胜概率。  
为此，在每次greedy move 后将value 回溯到之前的 move ，更准确地说，早期state 的value 会被更新到近于后期state 的value ，通过将后期state 的value 的一小部分加到前期state 的 value 来完成  
用 $s$ 表示greedy move 前的 state，用 $s'$ 表示 greedy move 后的 state，$V(s)$ 表示 $s$ 对应的 value ，那么：

$$V(s)\leftarrow V(s)+\alpha[ V(s')-V(s)]$$

α is a small positive fraction，称为 step-size parameter，影响学习的速率，这种方法叫做 *temporal-difference learning method*  
当 α 随时间适当下降时，该方法收敛；若 α 不随时间下降，策略也会缓慢改变下法，以更好应对当前对手  
进化算法使用一个固定策略进行多次对局，然后按照获胜次数来判断该策略的优劣，但忽略了对局过程中的每一步所包含的信息；  
使用value function ：则会在对局中，根据每一步造成的胜率变化来判断对应策略，比进化算法更加细化  
两者都是在策略空间中进行搜索，但学习value function 在信息利用上更有优势

#### Exercise：
*1.1：self-play*：最后会变成进化算法，因为两者都会根据对方的下法调整自己的策略，也就是对手固定的假设不存在，那么value function最终会寻找一个不针对任何对手的策略

*1.2：Symmetries*：可以用对称矩阵，加快学习速度，并不会改变其学习结果。如果对手不利用对称性，那么其在对称布局上的下法有可能不同，策略就不应该利用对称性来改变下法，还是应该对战中学到的策略来做应对，那么对称的state 就不会拥有对称的value，因为对手的下法是不同的。

*1.3：Greedy Play*：如果总是Greedy Play，那么策略就不会学到新知识，就可能错过其它胜率更高的下法，就不会变得更好，也不会变得更差，当然，一旦对手开始变化，那么策略就有可能变差

*1.4：Learning from Exploration*：
* With the step size parameter appropriately reduced, and assuming the exploration rate is fixed, the probability set with no learning from exploration is the value of each state given the optimal action from then on is taken, whereas with learning from exploration it is the expected value of each state including the active exploration policy. Using the former is better to learn, as it reduces variance from sub-optimal future states (e.g. if you can win a game of chess in one move, but if you perform another move your opponent wins, that doesn't make it a bad state) The former would result in more wins all other things being equal.
