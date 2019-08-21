---
layout: post
title: CHAPTER 8. Planning and Learning with Tabular Methods
date: 2019-08-12
tags: 强化学习
mathjax: true
---

本章将对强化学习方法做一个统一的描述，包括：  
基于模型的方法(*model-based*)，如动态规划与启发式搜索。  
无模型的方法(*model-free*)，如蒙特卡洛和时序差分。  

基于模型的方法主要依赖于 *planning* ，而无模型的方法则依赖于 *learning*  
这些方法既相似又不同，但它们的核心都是 值函数的计算。  
所有的方法都遵循：观察未来时间，计算树状值，然后使用值作为更新目标来优化逼近值函数。  

### 8.1 Models and Planning ###
一个模型可以告诉我们 agent 所需要用到的一切，来预测环境将会对 agent 的 actions 产生怎样的回应。  
给定一个 状态-动作 对，模型会给出 next state and next reward.  
随机模型意味着 next state and next reward 存在多种可能，各有其概率。  
分布模型(*distribution models*) 会给出所有这些可能及相应的概率的描述。  
采样模型(*sample models*) 则给出采样自这个概率中的其中一种可能。  
动态规划中用到的模型 $p(s',r|s,a)$ 就是一个分布模型。  
在第 5 章中的例子 blackjack 中用到的就是采样模型。  
分布模型显然比采样模型更强，因为我们可以用分布模型轻松得到采样。  
当然，在某些情形下得到一个采样模型会相对容易得多。  

模型可以用来模拟产生经历。  
给定一个初始状态和动作，采样模型会给出一个可能的转移，而分布模型会给出所有可能转移及概率。  
如果给定一个初始状态和策略，那么采样模型能够生成一整个 episode，而分布模型能给出状态转移树状图。  
我们可以说模型是用来模拟环境，或者说用来产生模拟经历的。  

*planning* 在这里表示一个计算过程，以一个模型为输入，通过与模型化的环境交互，来产生或优化一个策略。  

![figure_8_0_1](/assets/images/RL-Introduction/Chapter8/figure_8_0_1.png)

根据这个定义，在人工智能领域，有两个不同的方法来进行 planning  
*State-space planning*: 其包含了本书的方法，主要为通过在状态空间中搜索来得到最优策略或达到目标的最优路径，动作导致状态的转移，从而计算值函数。  
*Plan-space planning*: 改为在 plan space 中搜索。  
Plan-space 方法难以有效应用在随机序贯决策中(Russell and Norvig,2010)  

本章将描述所有 state-space planning 方法的一个统一的构造。  
其主要基于两点：  
1. 都将计算值函数作为优化策略的关键中间步骤；  
2. 计算值函数的方法，都是借由模拟经历来执行 update or backup.    

![figure_8_0_2](/assets/images/RL-Introduction/Chapter8/figure_8_0_2.png)

动态规划完美切合该结构，本章后面介绍的方法也都符合该结构。区别仅在于做何种更新、以如何顺序执行以及信息保留时间的长短。  

统一视角更加强调方法之间的关系。  
planning and learning 的核心都是对值函数的估计。  
区别仅在于 planning 用的是模拟经历，而 learning 用的是实际经历。  
当然，这个区别会导致一些其它的差别，比如如何分配性能以及如何灵活地产生经历。  
而统一的结构能够使我们共用两种方法的某些 ideas.  

本章另一个主题就是关于 planning 在 small, incremental steps 中的优点。  
这使得 planning 能够在任意时刻被打断或重定向，减小计算的损耗。  

### 8.2 Dyna: Integrated Planning, Acting, and Learning ###
当 planning 在线完成时，会在与环境的交互中产生一些有趣的问题。  
在交互过程中得到的一些新的信息可能会改变模型，由此与 planning 发生关系。  
因此在考虑当前乃至未来的状态与决策时，有必要定制 planning 的过程。  
而如果决策和模型学习都消耗较大的计算力，那么可用计算力就需要合理分配。  
本节引入的 Dyna-Q，是一个简单的结构，它集合了一个在线规划代理所需的主要功能。  
在 Dyna-Q 中出现的每个功能都是简单甚至破碎的形式。  
后续会精心设计这些功能，并做出权衡。  

在一个 规划代理 中，实际经历至少有两个功能：一是改善模型，二是直接优化值函数和策略。  
前者称作 *model learning* ，有时也叫作 *indirect reinforcement learning*;  
后者称作 *direct reinforcement learning*.  

![figure_8_0_3](/assets/images/RL-Introduction/Chapter8/figure_8_0_3.png)

使用模型的优点是 能够以更少的环境交互，更有限的实际经历 来达到更优的策略。  
而使用模型的缺点在于 复杂的模型设计，以及其与实际环境的偏差造成的影响。  
