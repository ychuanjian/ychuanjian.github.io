---
layout: post
title: CHAPTER 9. On-policy Prediction with Approximation
date: 2019-08-30
tags: 强化学习
mathjax: true
---

本章开始学习强化学习的函数逼近方法。  
实际上，函数逼近方法只是把第一部分中的表换成了一个由参数 ${\bf w} \in \Bbb R^d$ 决定的逼近函数而已。  
我们会将 $v_\pi (s)$ 写作 $\hat v(s, {\bf w})$ ,即在给定参数 ${\bf w}$ 下估计状态 $s$ 的值。  
$\hat v$ 可能是个关于 $s$ 的线性函数， ${\bf w}$ 是函数的特征矩阵；或 $\hat v$ 是个多层的神经网络，而 ${\bf w}$ 是各层连接向量的权重。  
通过调整 ${\bf w}$ ，能够得到任意的函数，且 ${\bf w}$ 的数量远小于状态空间的大小。  
当然，这样也导致改动一个权重会影响到多个状态对应的值，因此泛化虽然加强了学习的潜能，但也更加难以管理和理解了。  

神奇的是，拓展到函数逼近法上后，强化学习在观察受限问题(代理无法观察到所有状态)上也能起到作用了。  
如果 $\hat v$ 的参数化函数形式不允许估计值依赖于状态的某些方面，那么这些方面就像是不可观察的一样。  
本书中关于函数逼近法的所有理论结果在观察受限问题上是等同的。  
函数逼近做不到的是，用过去观察到的记忆来增强状态表示。  

### 9.1 Value-function Approximation ###
本书中的所有预测方法都被描述为更新到一个估计值函数，其将值在某些特定状态变换向一个关于该状态的 "backed-up value" 或者说是 *update target*.  
用一个定义式来描述这个独立的更新：$s\mapsto u$, $s$ 表示被更新的状态， $u$ 表示更新的目标。  
比如，在 Monte Carlo 中的 value prediction 就是 $S_t\mapsto G_t$； TD(0) 中就是 $S_t\mapsto R_{t+1}+\gamma \hat v(S_{t+1},{\bf w}_t)$,  $n$-step TD 就是 $S_t\mapsto G_{t:t+n}$.   
在 DP 中的 policy-evaluation update ， $s\mapsto \Bbb E_\pi[R_{t+1}+\gamma \hat v(S_{t+1},{\bf w}_t)|S_t=s]$. 是一个任意的状态 $s$ 被更新，而在其它情况中则是实际经历中碰到的 $S_t$ 被更新。  
将这个更新描绘成一个关于值函数的输入-输出行为，输入为状态 $s$ ，希望的输出是更新目标 $u$.  
让 $s$ 的更新泛化，影响到其它的状态，使得方法变得复杂灵活，在机器学习中将这种方式叫做**监督学习**(*supervised learning*), 而当输出是一个数值时，这个过程就叫做**函数逼近**(*function approximation*).  
函数逼近接收输入输出样例来得到关于它们的一个近似函数。  
将 $s\mapsto u$ 作为训练样例进行 value prediction 的函数逼近过程，再将得到的近似函数作为我们需要的 估计值函数。  
原则上，任何学自样例的监督学习方法都可以用于强化学习，包括人工神经网络、决策树以及各种多元回归方法。  
但并不是所有的函数逼近方法都适合强化学习，那些最为复杂的人工神经网络和统计方法均假设了一个静态训练集，然后在其上多次重复训练，这需要算法能够从不断增长的数据中高效地学习。  
另外，强化学习要求函数逼近方法具有处理非稳定的目标函数，如果方法不具备该能力，那它就不适用于强化学习。  

### 9.2 The Prediction Objective ($\overline{\rm VE}$) ###
这里为 prediction 的效果定一个明确的目标。  
在 tabular case 中不需要对 prediction 的质量做持续跟踪，是因为学习到的值函数会最终等于真实值函数，且每个状态的值各自独立，不会影响到其它状态。  
而在逼近方式中，状态之间的更新会彼此影响，而且不可能做到所有的状态的值都精确等于真实值。  
由于我们假设权重维度远低于状态空间的维度，所有想要某个状态变得精确，就会使得其它状态变得不准。  
因此必须确定哪些状态是最重要的，最需要达到精确情况的。  
由此定义了一个状态分布 $\mu (s) \geq 0,\ \sum_s\mu (s)=1$，用来表示对每个状态 $s$ 的误差的重视程度。  
对每个状态 $s$ 的误差，使用估计值  $\hat v(s, {\bf w})$  与真实值 $v_\pi(s)$ 的差的平方来表示，再用 $\mu(s)$ 作为其权重，得到一个目标函数，为**均方值差**(*Mean Squared Value Error*), 写作 $\overline{\rm VE}$:  

$$
\overline{\rm VE}({\bf w}) \doteq \sum_{s\in S}\mu(s)\Big[v_\pi(s)-\hat v(s, {\bf w})\Big]^2.  \tag{9.1}
$$

其中 $\mu(s)$ 经常用在状态 $s$ 上所花时间的比例，在 on-policy 训练中这称为 *on-policy distribution*.  
在 continuing tasks 中，on-policy distribution 是在策略 $\pi$ 下的稳定分布。  

![figure_9_0_1](/assets/images/RL-Introduction/Chapter9/figure_9_0_1.png)

$\overline{\rm VE}$ 并不能说一定就是强化学习的性能指标，我们要记得强化学习的最终目标---我们学习一个值函数也是为了找到一个更好的 **策略**，因此出于这个目标的话，是并不需要最小化 $\overline{\rm VE}$ 的。  
对 $\overline{\rm VE}$ 而言，理想的目标是找到一个**全局最优**(*global optimum*)的权重向量 ${\bf w^\ast}$, 使得对所有可能存在的权重 ${\bf w}$ ，都有 $\overline{\rm VE}({\bf w^\ast})\leq \overline{\rm VE}({\bf w})$.  
对于一些简单的函数逼近器(如线性函数)，这个目标是有可能达到的，但是对于复杂的函数逼近器(如神经网络与决策树)而言，这几乎是不可能的。  
而且，复杂的函数逼近器有可能会收敛于**局部最优**(*local optimum*)，也就是仅对那些存在于 ${\bf w^\ast}$ 周围的权重 ${\bf w}$ ，有 $\overline{\rm VE}({\bf w^\ast})\leq \overline{\rm VE}({\bf w})$.  
这在某些非线性情况下已经足够，甚至可能已经是最好的情形。  
然而还是有很多情况下，连收敛到局部最优都无法做到，哪怕是在某个最优的周围区域。  
有些方法实际上是发散的，它们的 $\overline{\rm VE}$ 取值最终会不受限的变化。  

### 9.3 Stochastic-gradient and Semi-gradient Methods ###
**随机梯度下降**(*stochastic gradient descent, SGD*) 是被最广泛使用的函数逼近方法，而且尤其适合 在线强化学习。  
在梯度下降方法中，权重向量是一个固定长度的列向量 ${\bf w}\doteq (w_1,w_2,\dots,w_d)^\top$, 而估计值函数 $\hat v(s,{\bf w})$ 则是一个在对所有的 $s\in S$ 上关于 ${\bf w}$ 可微的函数。  
在一个离散的时间序列上的每一步，对 ${\bf w}$ 进行更新，所以我们需要在每个时间 $t=0,1,2,3,\dots$ 的 ${\bf w_t}$.  
现在假设在每个时间步，都会观察到一个新的样例 $S_t\mapsto v_\pi(S_t)$，其包括一个状态 $S_t$ 和其在当前策略下的真实值。  
虽然在这种情况下得到了准确的真实值，但是对函数逼近器而言，由于有限的资源和方案，并不能轻易地做出更新。  
由于没有一个 ${\bf w}$ 能够使得所有状态---甚至所有的样例---达到精确值。  
因此必须将更新泛化到所有的状态中，要考虑那些在样例中没有出现的状态。  

现在假设样例是按照分布 $\mu$ 出现的，我们要做的是让 $\overline{\rm VE}$ 最小化。  
在该例中，去最小化观察到的样本的误差是一个不错的选择。  
SGD 在每个样例上，沿着使该样例的误差减少最多的方向，来对权重向量做一个微小的调整：  
$$
\begin{align}
{\bf w}_{t+1} & \doteq {\bf w}_t - \frac{1}{2}\alpha \nabla\Big[v_\pi(S_t)-\hat v(S_t,{\bf w}_t)\Big]^2 \tag{9.4}\\
& ={\bf w}_t+\alpha \Big[v_\pi(S_t)-\hat v(S_t,{\bf w}_t)\Big]\nabla \hat v(S_t,{\bf w}_t),  \tag{9.5}
\end{align}
$$

其中 $\nabla f({\bf w})$ 表示函数 $f$ 对变量 ${\bf w}$ 的每个分量的偏导，是一个列向量：  

$$
\nabla f({\bf w})\doteq \bigg(\frac{\partial f({\bf w})}{\partial w_1},\frac{\partial f({\bf w})}{\partial w_2},\dots,\frac{\partial f({\bf w})}{\partial w_d}\bigg)^\top. \tag{9.6}
$$

这个偏导向量就是 $f$ 关于 ${\bf w}$ 的**梯度**(*gradient*).  
SGD 中的梯度下降就是指在所有的时间步 ${\bf w}_t$ 上更新的值，都是与样例误差的方差的梯度呈负相关，这使得误差下降的速度最快。  
而 SGD 中的随机指的是样本的随机性。  
更新的量为一个微小值是因为我们要找的不是针对单个样本的零误差，而是针对全状态空间的低误差，因此通过每次前进一小步来寻求这样一个平衡点，而实际上的 SGD 常常使 $\alpha$ 逐步减小，使算法满足标准随机逼近条件，最终使算法收敛，SGD 能够保证收敛到局部最优。  

现在将在时刻 $t$ 的样例的输出定义为 $U_t\in \mathbb R$, 它不是确切的真实值，可以看做带噪声的真实值。不过还是可以用它代替真实值对权重做更新：  
$$
{\bf w}_{t+1}\doteq {\bf w}_t+\alpha \Big[U_t-\hat v(S_t,{\bf w}_t)\Big]\nabla \hat v(S_t,{\bf w}_t),  \tag{9.7}
$$

如果 $U_t$ 是一个对真实值的无偏估计，即 $\mathbb E[U_t|S_t=s]=v_\pi(S_t)$, 那么在常见的随机逼近条件也就是递减的 $\alpha$ 下，权重 ${\bf w}_t$ 能够保证收敛到局部最优。  
而一个状态的回报 $G_t$ 就是对真实值的一个无偏估计，所以就得到了蒙特卡洛的梯度下降算法：

![figure_9_0_2](/assets/images/RL-Introduction/Chapter9/figure_9_0_2.png)

而那些 bootstrapping estimate 则无法用在式 9.7 中，如 $n$-step returns $G_{t:t+n}$ 或 DP 的 target $\sum_{a,s',r}\pi(a|S_t)p(s',r|S_t,a)[r+\gamma \hat v(s',{\bf w}_t)]$ ,它们都包含了 ${\bf w}_t$，这意味着它们是有偏差的，因此无法给出一个真正的梯度下降方法。  
从另一个角度来理解，看公式 9.4 其更新用的 target 是一个梯度，它实际上与 ${\bf w}_t$ 的值是无关的。  
Bootstrapping methods 是不能真正地作为梯度下降的实例方法的。(Barnard, 1993)  
它们考虑了权重向量 ${\bf w}_t$ 在改变估计值上的作用，但是忽略了其在 target 上的作用。  
它们包含了一部分的梯度，因此称为 *semi-gradient methods*.  

semi-gradient 方法的收敛虽然并不鲁棒，但是在 线性函数情形下，其收敛性还是可靠的。  
当然，它们也有优点，如学习较快，能够在线连续学习等。  
一个典型的 semi-gradient 方法是 semi-gradient TD(0), 使用 $U_t\doteq R_{t+1}+\gamma \hat v(S_{t+1},{\bf w})$ 作为 target.  

![figure_9_0_3](/assets/images/RL-Introduction/Chapter9/figure_9_0_3.png)

*State aggregation* 是一种泛化函数逼近的方法，它将状态分组，每个组都有一个估计值(权重向量 ${\bf w}$ 中的一个分量)。  
一个状态的值被估计为它所在组的分量，当状态被更新时，那个分量也被独立更新。  
State aggregation 是 SGD 的一种特殊形式，对 $S_t$ 所在组的分量而言 $\nabla \hat v(S_t,{\bf w}_t)$ 为 1，而对其它分量而言则是 0.  

### 9.4 Linear Methods ###
函数逼近中最重要的一种特殊形式就是线性函数: 即 $\hat v(\cdot,{\bf w})$ 是关于 权重向量 ${\bf w}$ 的一个线性函数。  
对每一个状态 $s$， 都存在一个对应的实数向量 ${\bf x}(s)\doteq \big(x_1(s),x_2(s),\dots,x_d(s)\big)^\top$, 其维数与 ${\bf w}$ 相同。  
使用 ${\bf w}$ 与 ${\bf x}(s)$ 的内积来线性逼近状态值函数：

$$
\hat v(s,{\bf w})\doteq {\bf w^\top x}(s)\doteq \sum_{i=1}^dw_ix_i(s). \tag{9.8}
$$

这种情形下的近似值函数被称作**权重呈线性**(*linear in the weights*), 或简称**线性**(*linear*).  
此时，向量 ${\bf x}(s)$ 称为表示状态 $s$ 的**特征向量**(*feature vector*), 其每个分量都是函数 $x_i:\mathcal S \rightarrow \Bbb R.$ 的一个值。  
我们将一个特征视为这些函数之一的整体，我们将其值称为状态 $s$ 的一个特征。  
对于线性方法，特征是基函数，因为它们构成了所有近似函数集合的一个线性基础。  
构造一个 $d$ 维 特征向量来表示状态，与选择一个由 $d$ 个基函数组成的集合 是一样的。  
用 SGD 来更新 线性函数逼近非常自然，近似值函数关于 ${\bf w}$ 的梯度正是特征：

$$\nabla \hat v(s, {\bf w})={\bf x}(s).$$

因此，在线性状况下，通用的 SGD 更新简化成了一个特殊的形式：

$${\bf w}_{t+1}\doteq {\bf w}_t + \alpha \Big[U_t-\hat v(S_t,{\bf w}_t)\Big]{\bf x}(S_t).$$

因为简单，线性 SGD 在数值分析中很受欢迎，几乎所有类型的学习系统的所有有用的收敛结果都是线性函数逼近方法。  

特别地，在线性情形下，仅有一个最优点。 因此，任何能够保证收敛到局部最优的方法都自动升级为保证收敛到全局最优了。  

semi-gradient $TD(0)$ 算法在线性函数条件下也能收敛，但这并不是源自 SGD 的泛用结论，需要另行证明。  
权重向量并非收敛到全局最优，而是局部最优附近的某个点。  
我们需要更细致地了解这个重要情形，尤其是在 continuing case 下。  
在每个时间步 $t$ 时的更新如下：  

$$\begin{align}
{\bf w}_{t+1}& \doteq {\bf w}_t + \alpha \Big(R_{t+1}+\gamma {\bf w}_t^\top{\bf x}_{t+1}-{\bf w}_t^\top{\bf x}_{t}\Big){\bf x}_t  \tag{9.9} \\
& =  {\bf w}_t + \alpha \Big(R_{t+1}{\bf x}_t-{\bf x}_t({\bf x}_t-\gamma {\bf x}_{t+1})^\top {\bf w}_t\Big)
\end{align}$$

一旦系统稳定到某个状态，对于给定的 ${\bf w}_t$ ,能够直接写出其下个权重向量的期望：

$$
\Bbb E[{\bf w}_{t+1}|{\bf w}_t]={\bf w}_t + \alpha({\bf b}-{\bf Aw}_t), \tag{9.10}$$
其中: 
$${\bf b}\doteq \Bbb E[R_{t+1}{\bf x}_t]\in \Bbb R^d\quad \text{and}\quad {\bf A}\doteq \Bbb E\Big[{\bf x}_t({\bf x}_t-\gamma {\bf x}_{t+1})^\top\Big]\in \Bbb R^d \times \Bbb R^d  \tag{9.11}
$$

从式 $(9.10)$ 可以清楚：如果系统收敛，那么其必然收敛到向量 ${\bf w}_{\rm TD}$:  

$$\begin{align}
\qquad{\bf b}-{\bf Aw}_{\rm TD} & =0 \\
\Rightarrow \qquad \qquad \qquad {\bf b}& = {\bf Aw}_{\rm TD} \\
\Rightarrow \qquad \qquad \;\;\, {\bf w}_{\rm TD}& = {\bf A}^{-1}{\bf b}. \tag{9.12}
\end{align}$$

该值被称为 *TD fixed point*. 实际上 线性 semi-gradient $TD(0)$ 收敛于该点。  

在 TD 不动点上，$\overline{\rm VE}$ 被证明是位于最小可能误差之内的：  
$$\overline{\rm VE}({\bf w}_{\rm TD}) \leq \frac{1}{1-\gamma}\min_{\bf w}\overline{\rm VE}({\bf w}). \tag{9.14}$$

就是说，TD 法的渐进误差不会超过 $\frac{1}{1-\gamma}$ 乘上最小可能误差，这由蒙特卡洛法得到。  
而因为 $\gamma$ 常接近 $1$，该放大因子会显得特别大，所以在 TD 法的渐进性能上存在大量潜在损失。  
而另一面而言，TD 法相比于 蒙特卡洛，能够大大减少方差，且速度更快。  
因此，两者中那种方法更好，取决于问题和逼近的性质，以及学习过程持续多久。  

类似于 $(9.14)$， 对其它 on-policy bootstrapping methods 的边界分析也是适用的。  

对这些收敛结果而言，关键在于状态是根据 on-policy 分布来更新的。  
用其它更新分布，适用函数逼近的自助法可能发散到无穷大。  

#### 9.2：Bootstrapping on the 1000-state Random Walk ####
    跳过
    
在例 9.2 中使用的半梯度 $n$-step TD 算法是对表式 $n$-step TD 算法的一种很自然的拓展：

![figure_9_3_0](/assets/images/RL-Introduction/Chapter9/figure_9_3_0.png)

类似于式 $7.2$ , 算法中的关键公式为：

$$
{\bf w}_{t+n}\doteq {\bf w}_{t+n-1}+\alpha [G_{t:t+n}-\hat v(S_t,{\bf w}_{t+n-1})]\nabla \hat v(S_t,{\bf w}_{t+n-1}), \qquad 0\leq t < T, \tag{9.15}
$$

其中的 $n$-step return 可以归纳为：

$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots +\gamma^{n-1}R_{t+n}+\gamma^n\hat v(S_{t+n},{\bf w}_{t+n-1}), \qquad 0\leq t \leq T-n.  \tag{9.16}
$$

### 9.5 Feature Construction for Linear Methods ###
线性方法有收敛保证，且在实践中无论关于数据或是计算上都卓有成效。  
为任务选择合适的特征是给强化学习系统添加先验知识的重要途径。  
特征应当对应状态空间的某方面使其适合泛化。  

线性形式的一个限制是：它不能考虑到任何特征之间的互动关系，比如某个特征的状态仅在另一个特征某些情况下表现为优。  
比如在倒立摆实验中，摆杆的角速度较大，这一状态在摆杆位于下方 (角度值较大) 时为优，而当摆杆立直 (角度值较小) 时则是一个不好的状态，因为这会破坏立直。  
在特征分别编码表示的情况下，一个线性值函数无法表示这种情形。  

#### 9.5.1 Polynomials ####
许多问题的状态都被初始化表达为数值，这时，强化学习中的函数逼近法与插值(*interpolation*)和回归(*regression*)两类任务是相似的。  
在插值和回归中使用的多种特征也能够用于强化学习，多项式组成了最简单的一种特征。  
当然，这里讨论的基本多项式特征在并不如强化学习使用的其它种类的特征那么好用，但使用它来学习介绍是极好的，因为它们很简单且相似。  

举个例子，假设现在有两个数值状态 $s_1\in \Bbb R\ and \ s_2\in \Bbb R$ ，最简单地表示它们的方法就是用这二者组成一个二维状态向量 ${\bf x}(s)=(s_1,s_2)^\top$.  
显然，这样就无法顾及它们之间相互的作用，并且，假如两个状态同时为 $0$, 那么无论权重如何，其值只能是 $0$.  
当我们用 ${\bf x}(s)=(1, s_1, s_2, s_1s_2)^\top$ 来表示特征向量时，上面的两个限制都能被解除。  
甚至我们可以构造更高维的向量 ${\bf x}(s)=(1, s_1, s_2, s_1s_2,s_1^2,s_2^2,s_1s_2^2,s_1^2s_2,s_1^2s_2^2)^\top$. 来获取两个状态之间更加复杂的相互作用。  
这样的特征向量使得逼近器能够表示任意的关于状态的二次方程，即便逼近器所要学习的权重仍然是线性的。  
将 $2$ 个状态推广到 $k$ 个，就能够表示许多复杂问题中状态的相互关系了。  

假设每个状态 $s$ 对应 $k$ 个数值： $s_1,s_2,\cdots,s_k,\ with\ s_i\in \Bbb R.$   
那么对于这个 $k$ 维状态空间，每个 $n$ 阶多项式基础 特征 $x_i$ 可以写作：  

$$
x_i(s)=\prod^k_{j=1}s_j^{c_{i,j}}, \tag{9.17}
$$

其中每个 $c_{i,j}$ 是集合 $\{0,1,\cdots,n\}\ ,\ n\geq 0.$ 中的一个整数。这些特征组成了 $k$ 维 $n$ 阶多项式基础，其包含了 $(n+1)^k$ 种不同的特征。  

高阶多项式基底能使逼近器更准确的逼近复杂的函数。  
但是特征的数量也随 $n$ 阶多项式基底呈 指数增长(与状态空间的维度相关： $k$ 重)，所以必须为函数逼近器选择它的一个子集，而不是完全采用。  
这需要关于要逼近的函数本质相关的先验知识，以及一些自动化的挑选方法，以此解决强化学习本身的不稳定性和递进性质。  

#### 9.5.2 Fourier Basis ####
基于傅里叶级数的线性函数逼近方法。  
使用傅里叶级数和傅里叶变换，意味着一个函数近似已知，就能用更加简单的公式来得到其基函数权重；而且有了足够的基函数后，任何函数都能被精确地估计出来。  
强化学习中，需要被估计的函数是未知的，傅里叶基函数能够简单地表示它们，这在大部分强化学习问题中都是适用的。  

首先来看一维情形，具有周期 $\tau$ 的一维函数的常用傅里叶级数表示将函数表示为正弦和余弦函数的线性组合，每个函数都是周期的，周期平均地除以 $\tau$.  
但如果去近似一个非周期性的有界函数，就可以用这些傅里叶基底特征，将 $\tau$ 设置为区间长度。  
那个这个函数就看做是所设计的周期函数的一个周期。  

    傅里叶的内容略过啦

现在设计一个 $\tau=2$ 的函数，这样其半周期就落在区间 $[0,1]$ 内了，那么一维 $n$ 阶傅里叶余弦基底就包含了 $n+1$ 个特征：

$$x_i(s)=\cos (i\pi s), \quad s\in [0,1],\ for\ i=0,\cdots,n.$$

同样的，对于多维情形，也可以用类似的方法表示出其傅里叶余弦逼近器：

$$
x_i(s)=\cos(\pi {\bf s}^\top {\bf c}^i) \tag{9.18} \\
{\bf s}=(s_1,s_2,\cdots,s_k)^\top,\ with\ each\ s_i\in [0,1] \\
where\ {\bf c}^i = (c_1^i, \cdots, c_k^i)^\top,\ with\ c_j^i\in\{0,\cdots,n\}\\
for\ j=1,\cdots,k\ and\ i=1,\cdots,(n+1)^k.
$$

使用傅里叶余弦特征时，建议对每个特征使用不同的 step-size parameter.  
Konidaris, Osentoski and Thomas (2011) 建议设置：对 $x_i$，设其 $\alpha_i=\alpha/\sqrt{(c_1^i)^2+\cdots +(c_k^i)^2}$, 除非所有 $c_j^i=0$，那么可以直接使用 $\alpha$.  
在 Sarsa 中，傅里叶余弦特征的性能表现好过许多其它的基函数，包括多项式和径向基函数。  
但是，傅里叶特征对于非连续函数的表达存在困难，除非使用非常高频的基函数。  

如果状态空间维度较小的话，那么就可以使用 $n$ 阶傅里叶特征，这样特征选取就变成自动操作了。  
但对高维状态空间，就必须选择这些特征的一个子集了，这就用到了先验知识和一些选取技巧。  
在这方面，傅里叶基函数的优势是它可以简单地选取特征，只要将 ${\bf c}^i$ 向量设为状态变量间的相互作用，而限制 ${\bf c}^j$ 的值使得逼近器过滤掉高频部份的噪声。  
**另一方面，因为傅里叶特征在整个状态空间上非零，它们表示状态的全局属性，很难找到好办法来表示局部属性。**

图 9.5 对比了多项式基函数与傅里叶基函数在 1000-state random walk 上的表现。  
一般来说，不建议在 online learning 中使用多项式。

![figure_9_5](/assets/images/RL-Introduction/Chapter9/figure_9_5.png)

#### 9.5.3 Coarse Coding ####
