# 碎碎念

本笔记以及可能涉及的代码完全来自 : [**6.S184 An Introduction to Flow Matching and Diffusion Models**](https://diffusion.csail.mit.edu/)

本笔记由于是对照着原课程 50 多页的 notes 写的 , 更多是对原 notes 与视频课插入自己的理解 , 所以整个逻辑线会不停地反复缕 , 而展开会稍微马虎些 , 而到了一些式子的细节或理解问题上又会抠的非常细 , 总之建议只去当原课程的配餐 . 

且 Diffusion 类模型是我细致接触到的第一个生成模型 , 甚至学的时候对 GAN 还只是听说过的水平 , 因此整个笔记会出现大量主观臆断 , 自造词与不知所云 , 不要真把这东西当成什么权威笔记 . 

这里的版本是对原版本的修正 , 把里面的一堆自然自语用人类能理解的语言重写了一遍 , 顺便重新收获了许多新想法 .

# Part 0 全课程大观

这篇笔记聚焦于两种目前很广泛使用的 Generative AI Algorithms：Denoisiong Diffusion Models 与  Flow Matching , 二者虽然名字相差很大 , 却是本质同源的 .

核心观点：将噪音借助 ( 随机 ) 微分方程演变为输出的图片 /  视频等 , 与之前的 GAN 体系模型存在区别 . 

整体上的特点是容易实现但不太好理解 . 

为了展示方便 , 笔记中将以图片生成为例子 . 

6.S184 整个课程大致分为以下几个部分 : 
1. 将生成图片问题转化为概率分布采样问题 , 从而量化问题 . 
2. 介绍与构建微分方程 , 进而展示基于此的生成机制 . 
3. 寻找与构建训练目标 ，从而使模型学习可行 . 
4. 构建一个最原始的训练模型 . 
5. 改善模型结构来得到一个真正可用的模型 . 

但这里的 Part 分类会更随意一点 , 即使它们整体上的脉络与上面一样 .

# Part 0.99 从分布视角看实物 , 理型与采样过程

对于一个除了一堆图片什么都没有的数据集 ( 甚至没有标签 , 图片的内容挑选也不一定要有什么主题 ) , 我们仍然可以坚持认为它在反映它得以发生的某种意义 : 这种反映蕴含的意义会从完全有意义 ( 比如一个全是猫的数据集 ) 到 完全无意义 ( 完全随机图片 ) 之间取到某一个点 .

这种意义包含什么 ? 正常的想法一般会想到 , 是某种理型 . 我想要一个猫的集合 , 因而会将猫的图片放进去 , 且会将任何是猫 ( 至少是数据集构建者认为是猫 ) 的图片 , 挑一些放进去 . 如果我想要一个猫的集合 , 却为了方便一直选的是某一只猫的各个角度的图片 , 最终得到的反映理型则是这只猫而非全体猫的 . 而我可能需要一个随便的数据集 , 或者我的数据选取能力太烂 , 选到了一堆低质量图片 , 那么这个理型便也可能是某种不可名状的存在 . 也就是说数据集的存在本身在反映它的某种意志 . 如果我能得到它的这一意志的某种表示 , 也就能得到它的具化产物了 .  

但这个理解是存在进一步分析的空间的 : 采样的样本点来源就如此自然吗 ? 或许可以理解成 : 我们先在某一实物已经存在擅自决定定这一物体的本质 . 然后引入环境噪音 , 这种噪音引导着某物从最初一切都相同的地方生长成了独特的存在 , 再通过某种环境噪声被我们所观察 , 成为了我们所笃定的这一物体的一个样例 . 而前后两个噪音过程对于我们的观察行为本身可以等价成单个噪音过程 . 这种决定可以与物体所具有的 , 应当占据这一被决定本质的存在完全不同 , 但这种不同越大 , 越会使得噪音过程在某个视角中呈现出某种规律性 . 而如果这种决定在理想情况下与此物本质若完全相同 , 则噪音就会完全无规律 .

此时一种动态过程就采样中出现了 : 似乎是数据集本身除了理型意志 , 还包括者某种生成动态 ( 或者说前后者是相等的 , 或子集关系 ) , 指导随机经过某种过程流向这种理型 . 这种过程可以理解成一种场 , 而理型则是一个分布 . 一个从实物空间中随机取的点 , 被指定具有这一理型之本质后经受这种场的作用 , 最终停在某个位置 . 许多这样的点经历这样的过程停下来 , 它们的终点位置所反映的就是一个分布的一部分表现 , 也就是理型的一部分表现 . 而这些停下来的点被采样 , 构成数据集 .  

# Part 1 基于流思想的生成模型之概观

一个对于机器学习最基本的数学观点 : 图片等一切形式的采样数据都应该至少有一种向量表示 , 而这里可以认为它们都存在于 $\mathbb{R}^d$ 大小的特征空间中 .

首先是非条件生成 , 将数据集抽象为 $p_{data}$ , 也就是生成一个图片就是从 $p_{data}$ 中采样一个图片 .( 也就是说 , 我有一组狗的图片 , 我能拿到它 , 这个过程其实是从某个理想的图片集中采样来的 ( 选定 "狗 " ) , 也就是说 , 对于训练集 , 其承载着 $N$ 个从 $p_{data}$ 中采样的 $z_i$ , 即 $z_i \sim p_{data}$ )

在此基础上引入条件生成 : 假设 $y$ 是提示词 ( 比如 $y = \text{photorealistic image of a cat blowing out birthday candles}$ ) , 而这次让 $z$ 从 $z \sim p_{data}(\cdot|y)$ 中采样 . 

这里能看出来 , 无论是条件还是非条件生成 , $x$ 几乎只是个占位符 , 用来做初始诞生之物 , 填入系统生成的噪音 , 参与演变 , 最后成为图片输出 . 

称 $p_{data}(\cdot|y)$为 Conditional Data Distribution . 但一般而言其训练时不会将 y 固定后一直同一类别下训练 ( 这样与 Unconditional 情况就完全一样了 ) , 而是多种的 y 作为一个信息成为一数据集的部分 , 后面会给出 y 作为文本的处理方式 . 

PS . 后续推导可以得到 Condition 与 Uncondition 情况下很多公式只有一个常数项的区别 , 所以可以先用 Uncondition 进行推导 , 再拓展 . ( 但显然最后要的是 Condition 情况 . Unconditional 的情况更像是 Transformer 中不给 Encoder 内容让它直接输出 , 它会得到 "数据集中概率最高的句子" , 就像输入法在什么都不输 , 一直从头开始点推荐栏第一个得到的句子 , 不过这里对于图片生成模型更像是 " 在某个随机初始点下的数据集中概率最高的图片 " )

大概生成的过程 : 
- 将一批图片转化为一批向量 . 
- 给定图片集反映了一定规律 : 这些向量在空间中的位置的密度反映了一个理型的分布 , 根据此分布采样得到了这些向量 .
- 认为这个分布是由某个驱动场推动原空间点实现的 . 学习出这个场就能等价地得到分布 . 而这个过程 , 由于分布本身不可得 , 需要借用样本 . 我们初始化某个场 , 不断调整这个场的形状来使它逼近能生成样本集图片的效果 . 
- 学习完成后 , 定义一个随机分布 ( $p_{init}$ , 最典型为高斯分布 ) , 从中随机采样一个点 , 让它受场驱动 , 最后停下来的地方( 数学可证最后一定会停下 ) , 转回图片 , 就是输出 .

# Part 2 微分方程与随机微分方程

常微分方程 ( Ordinary Differential Equations , OEDs ) : 

定义一个向量场为 $u_t(x)$ , 定义域为空间中的每个空间与时间 ( 并约束时间为 $\left [0, 1\right]$ 区间 ) , 其为一个向量 $X$ 在 $t$ 时刻的导数 , 同时令 $X$ 在 $x_0$ 处开始运动 , 这样一来 $X$ 作为一个关于 $t$ 的函数 , 在每个 $t$ 的情况就是确定的 , 这也就是确定了一个轨迹 ( trajectory ) . 

而给定初始位置 $x_0$ 与指定时间 $t$ , 可以得到在 $t$ 这一时刻 $X$ 的位置 , 这是一个函数 , 称为 $flow$ , 记为 $X_t = \psi_t(X_0)$ . 

$\text{Vector fields define ODEs, whose solutions are flows}$ 这句话可以解释三者的关系 . 

而 $\psi_t(x)$ 在以下条件下 , 是可以保证存在且唯一存在的 : $u$ 可微且其导数有界 . ( 且其与其逆函数是连续可导的 ) . 而这个条件在 Machine Learning 中是很容易达到的 .

抽象为最原始的形态 , 即是 $dX_t = u_t(X_t)dt\quad\quad X_0 = x_0$

一个实际例子，可以举 $\psi_t(x_0) = e^{-\theta t}x_0 $ 是 $u_t(x) = -\theta t$ 的 flow

但在实际计算中 , 算导数时用符号运算是不可能的 , 数值计算是主流 . 这里更新选择采用离散的方法 . 这种计算方法有非常多中 . 比较常用的一个为 Euler method , 更新公式为 $X_{t+h} = X_t + hu_t(X_t) \space (t = 0,h,2h,\dots,1-h)$ , 其中 $h$ 作为 $\Delta t$ 进行简单的导数式的步进 . 而为了和后面的代码统一 , 我们也称这些方法为 Simulator , 取其模拟微分方程运算之意 . 

为证明还有别的方法 , 这里举例 Heun's method :  
$$X'_{t+h} = X_t + hu_t(X_t)$$ $$X_{t+h} = X_t + \frac{h}{2}(u_t(X_t) + u_{t+h}(X'_{t+h}))$$

注意这里对 Simulator 的定义是非常宽泛的 : 只要求其且提供一个针对一个小的时间变化进行的步进操作 , 以及基于此对一串时间段切割进行的模拟演化操作 . 

也就意味着其甚至能允许不均匀切割时间的方法 , 因此要注意哪里是这个模块的本质 , 哪里是不同策略间的选择 .

这种通过训练的到的参数构建向量场 , $x$ 从初始分布中用 ODE 随机采样游走停下 , 便是 flow model ( **注意不是flow matching , 这两个不是一个东西** ) , 而目标是让游走终刻得到的 $\psi^\theta_1(X_0) \sim p_{data}$ : 这个演变在训练好的参量下 , 能使得输入的随机初始值在演变终点的结果服从 $p_{data}$

而 SDE ( Stochastic Differential Equations , 随机微分方程 ) 在 ODE 基础上加上了一个 Brownian motion ( 又名 Wiener process , 因此简记其项为$W$ ) . 其定义为 : 

$dX_t = u_t(X_t)dt + \sigma_tdW_t\quad\quad X_0 = x_0$

Brownian motion 属于一种 stochastic process , $W = (W_t)_{0\le t\le 1}$ , $W_0 = 0$ , 其对于时间 $t$ 的轨迹连续 , 满足 :
1. Normal increments : $W_t-W_s\sim \mathcal{N}(0,(t-s)I_d)\space0\le s\le t$ , 也就是说其增量的方差呈线性增长 .
2. Independent invrements :  对顺序时间列 $t_i$ , 有 $W_{t_1}-W_{t_0},\dots,W_{t_n}-W_{t_{n-1}}$ 之间相互独立 . 
   
这样一来 Brownian process 可以以 $W_{t+h} = W_t + \sqrt{h}\epsilon_t,\space \epsilon \sim \mathcal{N}(0,I_d)$ 的形式更新 . 

题外话 : 纯粹的 Brownian Motion 是分形 , 这个证明思路是其缩放后仍是自己 ( 或按时间缩放尺度等于其自身变化后整体缩放尺度 ) , 很有意思

而由于引入随机量 , 取随机行为是离散的 , 整个 SDE 也就要用不含导数的方式描述了 . 以 Euler method 为基础 , 可以写出 : $$X_{t+h} = X_t + hu_t(X_t) + \sigma_t(W_{t+h}-W_t) + hR_t(h)$$

( 其中布朗运动本身前后差已经引入了标准差为 $\sqrt h$ 的缩放信息 , 故 $W$ 参与的项不用乘 h , h 相当于 dX . 而 $R$ 为 $h$ 的无穷小 , 实际写代码时可以忽略 . )

进而可以写成如下的普遍形式 :
$$dX_t = u_t(X_t)dt + \sigma_tdW_t$$
$dt$ 前系数 ( 这里是 $u_t(X_t)$ ) , 被称为 Drift Coefficient , $dW$前系数 ( 这里是 $\sigma_t$ ) 被称为 Diffusion Coefficient .

( 注意这个式子只管运动 , 运动成什么样子 , 波动到什么幅度都不管 , 这也为后面修正它为可用的式子留了空间 . )

一个特殊情况：drift 为 $-\theta$ , diffusion为 $\sigma$ , 这种情况被称为 Ornstein-Uhlenbeck (OU) process

而之前的式子经过进一步化简 , 得到 Euler-Maruyama method : $X_{t+h} = X_t + hu_t(X_t)+\sqrt{h}\sigma_t\epsilon_t$ , $\epsilon \sim \mathcal{N}(0,I_d)$ 为单位噪音

因为只有引入新的随机项的差别 , 可以照着 Flow Model 造出来几乎一样的 Diffusion Model

# Part 2.2 简单补充介绍些具有名字的存在

由于这里大部分定义只是定下必要的基本形式 , 而大部分具体实现方式是放开的 , 因此在各个部分都有出现有自己名字的方法 . 尤其是 Lab 中初见会比较晕 . 这里以 Lab 代表的分层思路为主干 , 补充说明一下 . 

首先 , 这其中涉及的原始概念有 : 
- ODE / SDE : 基础方程的组分 . 不同种类的过程需要提供各级的漂移参数与扩散参数
- Simulator :  离散化演变过程的组件 , 需要提供根据时间刻具体步进的策略 , 称为 method 也无不妥 .  
- Density : 理想中的目标分布 , 描述它可能的在可解析的情况下能给出 Score Function 等的性质 
- Sampleable : 理想中的目标分布 , 描述它不一定可解析 , 但应该可采样的性质 . ( 与 Density 相当于一个分布的并列的两大性质 )
  
## SDE
### Brownian Motion
Brownian Motion 本身也可以被视为一种特殊的 SDE , 有
$$ dX_t = \sigma dW_t, \quad \quad X_0 = 0.$$

也就是说 Drift Coefficient $u_t$ 为 0

### Langevin Dynamics
这是在这个背景下真正有应用价值的 SDE 构造策略 .
$$dX_t = \frac{1}{2} \sigma^2\nabla \log p(X_t) dt + \sigma dW_t.$$

如果没看过后面的章节 , 只需要目前知道有这样一种构造策略 , 然后就可以跳到下一部分 .

但如果已经看过后面的章节 , 会发现它和后面真正用的 $\mathrm{d}X_t = \left [u_t^{target}(X_t) + \frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\right ]\mathrm{d}t + \sigma_t\mathrm{d}W_t$ 非常像 , 这里重点分析下二者差别 . 

最主要差别有两点 : 前者 $p$ 与 $t$ 无关且不含有 $u$ 项 . 第一项特点意味着前者的 $p$ 完全就是目标分布 , 也就是我们还没开始拟合就能知道目标分布的解析式级别的表示 , 这显然不可能 . 但这也暗示了一个信息 : $\frac{1}{2} \sigma^2\nabla \log p(X_t) dt$ 它可以提供一个指向分布的移动量 . 

为了有解析形式 , 我们引入了 Gaussian Probability Path 来想办法提供一个解析式的路 , 而这需要它有一个时间的演进 , $p$ 变成 $p_t$ . 也就是 Score Function 会提供该时刻的一个好的方向驱动 . 而在引入 ODE 的 $u$ 来进一步引导主方向正确 . 这种解释实际上与后面的解释是有不同的 , 或许可以多角度理解 ? 

### Ornstein-Uhlenbeck Process ( OU Process )
其选择设置漂移系数为常量 , 得到 
$$ dX_t = -\theta X_t\, dt + \sigma\, dW_t$$ 
不难发现这个可以将点以指数的趋势推回原点 . 

在读完其余部分 ( 至少理解大概 Lagevin Dynamics 是什么后 ) , 用一下事实加强对感觉 : 

**当取目标分布为恒常 $p(x) = \mathcal{N}(0, \frac{\sigma^2}{2\theta})$ 时 , 其 Langevin Dynamics 就是 OU Process**

这可以通过证明 $-\frac{2\theta}{\sigma^2}x = \nabla \log p(x)$ 得到 . 但这也为上面 " 推回原点 " 提供了更具体的解释 : 当时间趋向无穷时 , 分布会趋向回到一个原点附近的正态分布 . 

## Simulator : 
### Euler Simulator : 
应用于 ODE , 更新选择 $$X_{t + h} = X_t + hu_t(X_t)$$ 最直观的选择 . 

### Euler-Maruyama Simulator : 
应用于 SDE , 更新选择 $$X_{t + h} = X_t + hu_t(X_t) + \sqrt{h} \sigma_t z_t, \quad z_t \sim N(0,I_d)$$ 是 Euler Method 的基础拓展 . 

# Part 3 训练目标函数的寻找与处理

作为学习，需要训练的模型来自 $u$ , 而这里优化选择 MSE , 也就是应该有 :
$$\mathcal{L}(\theta) = \left \| u_t^\theta(x) - u_t^{target}(x) \right \|^2$$

( 有的地方也会写 $u_t^{target}$ 为 $u_t^{ref}$ )

但现在的问题是：没有现成的 $u_{target}$ , 只有从目标分布里得到的图片。

事实上，虽然是要找到 $u_{target}$ , 但很明显它的表达式里会有不可得到的 $p_{data}$ 的参与 , 这是推导前就能意识到的 . 所以找到 $u_{target}$ 表达式只是部分目标 , 另一部分是如何用采样数据代替表达式中的 $p_{data}$ .

以下是一种相对简单且广泛使用的策略 : 

为了进一步描述 $x$ 从初始到最终所属分布的演变过程 , 类似 flow 从确定的运动轨迹之类的应用被拓展成概率分布版本 , 我们可以定义 Probability Path : 对于一族 $p$ , 其以 $t$ 为参数 , 反应 $t$ 时间时的 $x$ 的分布情况情况 ( 的确似乎 $x$ 在取样后就与其所属分布 $p_0$ 无关了 , 但要注意这些 $x$ , 在大量取样后于 $t$ 点在空间中也会存在分布密度 , 也就是说它在 $t$ 时刻也会反应一个分布 , 这就是所指的 $t$ 时刻分布 )

而这个 Path 可以分为两种 : Conditional ( Interpolating ) Probability Path 与 Marginal Probability Path . 前者固定某个数据样本z研究 $p(\cdot|z)$ , 后者研究整个分布的情况$\int p_t(x|z)p_{data}(z)dz$ ( 直观来讲就是前者是初始化一堆点后只受到这一张图的引力 , 后者是这些点会受到所有样本点的引力 )

首先考虑固定某个 $z$ ( 即选定某个样本 , 让图片去拟合那个单张图片的情况 ) .
这种情况在理想状态下显然有 $p_0(\cdot|z) = p_{init}$ 以及 $p_1(\cdot|z) = \delta_z$

其中 $\delta_z$ 表示 " Dirac delta distribution " , 其某种意义上是最简单的分布 : 有且必然有 $x = z$ , 即 $p(x = z) = 1$ , $p(x \ne z) = 0$ , 也就是说最后 $p_1$ 其实表示的是生成 $z$ 所代表的这张图片本身 .

而将 $z$ 积起来消去 , 分析 $x$ 本身就会得到 Margenal Probability Path , 二者的关系与常规的条件概率与边缘概率基本一致 . 而 $x$ 与取样 $z$ 存在关系 , $z$ 与数据集存在关系 , 故而可以直接搭建 $x$ 与数据集间的概率关系 : 
$$z \sim p_{data},\space x\sim p_t(\cdot|z)\space \Rightarrow x\sim p_t$$ $$p_t(x) = \int p_t(x|z)p_{data}(z)dz$$

也就是说 , 现在可以将 " $t$ 时刻 $x$ 位于空间中某特定点 " 这一状态理解为 : 
1. 被初始化后借助 $u$ 与 $W$ 驱动过去
2. 从 $p_t(x)$ 中采样得到
   
而这个 Marginal Path $p_t$ , 无论中间过程如何 , 都应该至少保持两个要求 : $p_0 = p_{init}$ , $p_1 = p_{data}$

注意这里 : Marginal Path 是从定义层面上不能变的 , 但其实没人管 $p_t(x|z)$ 是什么东西 , 也就是说我们可以自己选择 Conditional Probability Path 的形式 .

而如果这里这两种移动方式可以以某种方式等效过去 , 就意味着有能构建的方程 . 

接下来便用 $p_t$ 进行 $u_t^{target}$ 的构建 ( 先是基于 flow model , 用 ODE 构建 )

首先仿照 Conditional Probability Path , 对任意从样本集中采样的 $z$ , 我们希望能有 $u^{target}_t(\cdot|z)$ 来表示一个能使 $X$ 的概率路径服从 $p_t(\cdot|z)$ 的 Conditional Vector Field .

而在此基础上 , 设法消去 $z$ 得到的 $u^{target}_t(x)$ 被具体定义为 $$u^{target}_t(x) = \int u^{target}_t(x|z)\frac{p_t(x|z)p_data(z)}{p_t(x)}dz$$

( 这个其实是可以从上面的原始语言定义中推出来的 . 后面会给出为什么是这个式子的证明 )

( 注意以上那个式子虽然初见比较莫名其妙 , 但也不是不能猜着初步理解下 : 概率项可以整理为 $p_{data}(z|x)$ , 进而这其实理解为 : 给定一个点 $x$ , 它对各个 $z$ 都有一定的收敛趋势 , 而这种趋势的强弱由 $p_{data}(z|x)$ 表现出来 . 这种趋势作为权重 , 去和这个点造成的引力 $u(x|z)$ 做的加权和 , 即是整体上 $u^{target}$ 在 $t$ 时刻的值 . )

而由这个定义内涵可以反推出 : 满足 $\frac{d}{dt}X_t = u^{target}_t(X_t)$ 的 $X$ , 服从 $p_t$

这个定理被称为 Marginalization Trick . 这实际上提供了很大思路突破 , 因为它相当于在说用Conditional 情况间接推 Marginal 情况是可行的 , 而 Conditional 情况是相对好算些的 , 这里以 Gaussian Conditional Probability Path 举例 . 

回忆上面对 $p_t$ 与 $p_t(x|t)$ 的灵活构造 , 基于此可以设计一种既特殊又比较质朴的 Conditional Probability Path : Gaussian Conditional Probability Path , 它在应用中比较多，尤其是 Denoising Diffusion Models 是专指用它的模型的 . 

( 由于其影响力大而且好算 , 接下来的分析也会基本用它进行 . )

首先引入两个 Noise Sheduler $\alpha_t,\beta_t$ : 其可以视为 $t$ 的函数 , 且对 $t$ 单调 , 满足 $\alpha_0 = \beta_1 = 0$ , $\alpha_1 = \beta_0 = 1$

进而令 : 
$$p_t(x | z) = \mathcal{N}(\alpha_tz,\beta^2_tI_d)$$
也就是
$$x = \alpha_tz + \beta_t\epsilon \sim p_t$$
( 很容易验证它满足两个对pt的要求 )
( 注意这里两个参数是认为规定好的函数 , 而不是需要学的 , 一种比较简单的设定 : $\alpha_t = t$ , $\beta_t = \sqrt{1-t}$ )

对于这个设定给的非常清晰的函数 , 是能直接算出来 $u_t^{target}(x|z)$ 的 ( 而其它大多情况下解 $u_t^{target}(x|z)$ 本身都很困难 , 更不用说后面的积分 ) : 
$$u_t^{target}(x|z) = \left ( \dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right )z + \frac{\dot\beta_t}{\beta_t}x$$

大概证明过程 : 

首先既然要求 $u^{target}$ , 也就意味着 flow 为 $\psi^{target}_{t}(x|z) = \alpha_tz + \beta\epsilon$

而根据定义 , 对其关于 $t$ 取微分得 $$\frac{d}{dt}\psi^{target}_{t}(x|z) = u^{target}_{t}(\psi^{target}_{t}(x|z)|z)$$ 
( 原定义为 : $\frac{d}{dt}\psi_t(x) = u^{target}_{t}(\psi_t(x)|z)$ , 没用 $u^{target}_{t}(x)$ 是因为这里一直在讨论的是已经取号 $z$ 的情况 )

进而有 $$\dot{\alpha_t}z + \dot{\beta_t}x = u^{target}_t(\alpha_tz+\beta_tx|z)$$

对其作换元 $y = \alpha_tz+\beta_t x)$ , 得到 $$\dot{\alpha_t}z + \dot{\beta_t}\left ( \frac{y - \alpha_t z}{\beta_t} \right ) = u^{target}_t(y|z)$$

整理并换符号回 $x$ , 得到 $$u_t^{target}(x|z) = \left ( \dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right )z + \frac{\dot\beta_t}{\beta_t}x$$

而对 Marginalization Trick的证明需要用到 Continuity Equation ( 而它本身里主题比较远 , 这里就只写理解 , 不具体证明了 . )

Continuity Equation : $$\partial _t p_t(x) = - \text{div}\left(p_tu_t^{target}\right)(x)\quad  ( \partial _t p_t(x) = \frac{\mathrm{d} }{\mathrm{d} t}p_t(x) )$$
( 这里 $\text{div}$ 求散度得到的是标量 , 左边因为是对 $t$ 求偏导所以也是标量 ) 
( $div(p_tu_t^{target})(x)$ 与 $div(p_t(x)u_t^{target}(x)) $ 是一个意思 )
( 就像正常来说用 $x$ 增加为正方向 , 这里用净流入 ( 散度取负 ) 来表示流量变化的正方向 , 从而用约定统一方向 , 而证明等式本身其实也能证明这种约定合理性 ( 同时等式被推出给出了这种约定的必然性 , 总而言之是互推的 ) )
( 某对于一个分布 , 可以认为 $u_t^{target}$ 是空间某一点在某一时刻概率密度向各个维度的流失速度 , 而取某一个极小的单位空间与单位时间 , $p_t(x)dtdV$ 即是其在这一点的概率量 ( 而不仅仅是概率密度量 ) , $p_t(x)u_t^{target}(x)dt$ 则是这一点的概率量向各个方向的流失量 . 它们在全体维度上求和就是该点单位空间概率量的变化量 $\partial_tp_t(x)dV$ 的负值 . 建立等号 , 化简就能得到 Continuity Equation )

现在是基于以上公式对 Marginalization Trick 的证明 : 思路是将给定定义的 $u$ 回代已确定的 $p$ 表达式 , 回推出预设条件下必然成立的 Continuity Equation . 

首先出发点是没有异议的 Marginal Probability Path的表达式 : 
$$p_t(x) = \int p_t(x|z)p_{data}(z)\mathrm{d}z$$
然后两侧对 $t$ 求偏导 : 
$$\partial_t p_t(x) = \partial_t \int p_t(x|z)p_{data}(z)\mathrm{d}z$$
因为积分与偏导作用的变量不同 , 位置可以互换 : 
$$\partial_t p_t(x) = \int \partial_t p_t(x|z)p_{data}(z)\mathrm{d}z$$
接下来对 $\partial_t p_t(x|z)$ 部分用 Continuity Equation替换 : 
$$\partial_t p_t(x) = \int -\mathrm{div}(p_t(\cdot|z)u_t^{target}(\cdot|z))(x)p_{data}(z)\mathrm{d}z$$
同样，求散度作用对象是 $x$ , 积分作用对象是 $z$ , 可以互换位置 : 
$$-\mathrm{div}\left (\int p_t(x|z)u_t^{target}(x|z)p_{data}(z)\mathrm{d}z\right )$$
接下来在积分号内外分别乘 / 除一个 $p_t(x)$
$$-\mathrm{div}\left (p_t(x)\int u_t^{target}(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)}\mathrm{d}z\right )$$
接下来代入 $u_t^{targer}(x)$ 的定义 ( 或者说与 Continuity Equation对比 ) , 得到结果 , 得证 :
$$\int u_t^{target}(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)}\mathrm{d}z = -\mathrm{div}\left ( p_tu_t^{target}\right )(x)$$

接下来可以考虑 Diffusion Model 与 SDE 情况了 . 

首先定义 Marginal Score Function , 其输入 $p_t(x)$ , 输出 $\nabla\log p_t(x)$

之前只是提到了一个最宽泛的SDE : $\mathrm{d}X = u\mathrm{d}t + \sigma\mathrm{d}W$
而为了让它满足约束 : $X_t \sim p_t$ ( 注意这是对 $t$ 始终的而不是只限于 $0$ 与 $1$ 的 ) , 需要将 SDE 构建为 : 
$$\mathrm{d}X_t = \left [u_t^{target}(X_t) + \frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\right ]\mathrm{d}t + \sigma_t\mathrm{d}W_t$$
( 也就是引入一个修正项来抹掉随机项对分布的扩散影响 )
( 注意随机项在采样规则上是一个确定的东西 , 而采到的结果是随机的 , 同理分布也是采样有随机性 , 规则表达式本身无随机性 , 所以可以用精确的式子消去所谓的随机影响 , 或者说这种消除是在 ) 
（ 而尽管在构建SDE时修正了随机项，真正生成点后运动时是还是有随机项参与的，这也就是说运动轨迹仍是有随机参与的 . 具体来讲 , 如果只是单纯引入随机漂移 , 其在时间上的积累会使得采样点愈发分散 , 但我们想要的是它在运动时有些许飘动 , 但最终落点在经过充分采样后仍能得到不引入随机漂移的原分布 . ）
( 而将 $x$ 换为 $x|t$ 后对 Conditional Probability Path式子仍然成立 )
( 注意其与流的关系 : $u$本身可以确定一个确定的方程 $\psi_t(x)$ , 而 $X_t$ 是由其与随机过程混合得到的 , 因此可以在 OSD 中随意替换二者 , 但在 SDE 中不行 )
( 对其的证明在后面有给出 )

接下来先看看式子里的 Score Function 部分

完全类比前面的处理 , 算 Probability Path 时 , 先考虑 Conditional 情况会更简单些 , 而 Marginal 情况可以由前者做一些处理得到

比如 Gaussian Path 里 , 可以根据定义式直接计算出 :
$$\nabla \log p_t(x|z) = \nabla \log \mathcal{N}(x;\alpha_tz,\beta_t^2I_d) = -\frac{x-\alpha_tz}{\beta_t^2} $$

而一般地 , 有 $\nabla \log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)}$ ( 根据定义与一次链式法则可以得到 ) 
进而得到 
$$
\begin{aligned}
\nabla \log p_t(x)  &{=} \frac{\nabla \int p_t(x|z)p_{data}(z)\mathrm{d}z}{p_t(x)} \\
& {=} \frac{\int \nabla p_t(x|z)p_{data}(z)\mathrm{d}z}{p_t(x)} \\
& {=} \int \nabla \log p_t(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)}\mathrm{d}z 
\end{aligned}
$$

而对 SDE 构建式的选取的证明需要用到 Continuity Equation的推广 : Fokker-Planck Equation

首先定义 Laplacian Operator $\Delta$ 为 : $$\Delta w_t(x) = \sum^d_{i=1}\frac{\partial^2}{\partial x_i^2}w_t(x) = div(\nabla w_t)(x)$$

则对通用的 SDE 表达式 
$$\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t$$
当且仅当引入 Fokker-Planck Equation $$\partial_tp_t(x) = -\mathrm{div}(p_tu_t)(x) + \frac{\sigma^2_t}{2}\Delta p_t(x)$$ 修正 Drift Coefficient 后才能做到 $X_t \sim p_t$

有前面证明的基础后这里相对比较简单，核心只是造出来已知方程左右需要的项 . 

$$
\begin{aligned}
\partial_{t} p_{t}(x) & {=}-\operatorname{div}\left(p_{t} u_{t}^{\mathrm{target}}\right)(x) \\
& {=}-\operatorname{div}\left(p_{t} u_{t}^{\mathrm{target}}\right)(x)-\frac{\sigma_{t}^{2}}{2} \Delta p_{t}(x)+\frac{\sigma_{t}^{2}}{2} \Delta p_{t}(x) \\
& {=}-\operatorname{div}\left(p_{t} u_{t}^{\mathrm{target}}\right)(x)-\operatorname{div}\left(\frac{\sigma_{t}^{2}}{2} \nabla p_{t}\right)(x)+\frac{\sigma_{t}^{2}}{2} \Delta p_{t}(x) \\
& {=}-\operatorname{div}\left(p_{t} u_{t}^{\mathrm{target}}\right)(x)-\operatorname{div}\left(p_{t}\left[\frac{\sigma_{t}^{2}}{2} \nabla \log p_{t}\right]\right)(x)+\frac{\sigma_{t}^{2}}{2} \Delta p_{t}(x) \\
& {=}-\operatorname{div}\left(p_{t}\left[u_{t}^{\mathrm{target}}+\frac{\sigma_{t}^{2}}{2} \nabla \log p_{t}\right]\right)(x)+\frac{\sigma_{t}^{2}}{2} \Delta p_{t}(x)
\end{aligned}
$$

此时对比一下就能得到修正后的 Drift Coefficient

进而绕来绕去我们得到了借助 $u_t^{target}(x|z)$ 构造 $u_t^{target}$ 的方法 , 并且通过选择其采样方式使得解析地写出它在取个例情况下的 $u_t^{target}(x|z)$ 的计算式成为可能 . 在下一部分能看到 , 虽然由于 $\int$ 太难算 , 我们仍然没法将 $u_t^{target}$ 直接用来计算 , 但是借助这里的计算成果 , 我们可以将计算它的过程绕开得到真正可行的学习过程 . 

# Part 4 寻找可行的学习方式

复习：一直以来在训练部分的目标是：
$$\mathcal{L} = \|u_t^{\theta} - u_t^{target} \|$$

现在可以开始考虑如何训练了 . 接下来在 ODE 与 SDE 中 , 我们从训练 MSE 出发推导出的策略为 Flow Matching 与 Score Matching . 而取 Gaussian Probability Path 这个特殊情况导出来的为 Denoising Diffusion Models

首先来看 Flow Matching

先区分概念 , 称 Flow Model为 $$X_0 \sim p_{init},\quad \mathrm{d}X_t = u_t^{\theta}(X_t)\mathrm{d}t$$

为了后续方便，记 $\mathrm{Unif} = \mathrm{Unif}_{[0,1]}$

首先定义 Flow Matching Loss ( FM ) :
$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta) & =\mathbb{E}_{t\sim\mathrm{Unif},x\sim p_t}[\|u_t^{\theta}(x)-u_t^{\mathrm{target}}(x)\|^2] \\
 & =\mathbb{E}_{t\sim\mathrm{Unif},z\sim p_{\mathrm{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u_{t}^{\mathrm{target}}(x)\|^{2}]
\end{aligned}
$$

这个过程似乎是在描述这个训练过程：
1. 取样一个随机时间刻
2. 从数据集中取样一个 $z$
3. 从 $p_t(\cdot|z)$ 中取样一个 $x$
4. 算 $u^{\theta}_t(x)$

( 注意从前面的各种分析可以知道对 $p(·|z)$ 的约束是人为确定的 , 是完全的 . 是可以写出解析式而不含有待学习参量的 . )


但问题在于 : 回忆 $u^{target} = \int \cdots$ , 这个积分几乎没有解析形式 ( 很好理解 , 即使随便写个积分式 , 有解析解的也不多 ) , 即使取最简单的高斯情况 , 在 $u^{target}(x|z)$ 基础上进一步积分处理也是非常麻烦的 .  

但也能注意到 , 在 Conditional 情况下 , $u_t^{target}(x|z)$ 是可以有解析解的 ( 处理它的全程都没有牵及积分 ) , 对比先前定义可以先令 Conditional Flow Matching Loss ( CFM ) 为 :
$$\mathcal{L}_{\mathrm{CFM}}(\theta) =\mathbb{E}_{t\sim\mathrm{Unif},z\sim p_{\mathrm{data}},x\sim p_{t}(\cdot|z)}[\|u_t^{\theta}(x)-u_t^{\mathrm{target}}(x|z)\|^2]$$

( 注意无论有没有 Conditional , 都只是在针对 $u^{target}$ 展开讨论 , $u^theta_t(x)$ 本身没有涉及到 Conditional 的操作 )

而此时可以得到一个重要发现 : 回归 CFM Loss 的同时 FM Loss 也会被回归 . 这是因为可以证明 :
$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathcal{L}_{\mathrm{CFM}}(\theta) + C \\
\nabla\mathcal{L}_{\mathrm{FM}}(\theta) = \nabla \mathcal{L}_{\mathrm{CFM}}(\theta)  
\end{aligned}
$$

( 待训练参数在两种 Loss 中共用且相同 , 而二者梯度相等 , 所以可以替代地训练 )

而证明过程其实是将范数展开得到的 :

先看 $\mathcal{L}_{\text{FM}}(\theta)$ . 首先是定义式 :

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[ \|u_{t}^{\theta}(x)-u_{t}^{\text{target}}(x)\|^{2}]$$

然后按照 $\| a - b\|^2 = \|a\|^2 - 2a^Tb + \|b\|^2$ 展开 : 

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},x \sim p_{t}}[\|u_{t}^{\theta}(x)\|^{2}-2u_{t}^{\theta}(x)^{T}u_{t}^{\text{target} }(x)+\|u_{t}^{\text{target}}(x)\|^{2}]$$

利用期望的线性性拆开 : 

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},x \sim p_{t}}\left[\|u_{t}^{\theta}(x)\|^{2}\right]-2\mathbb{E}_{t\sim\text{Unif},x \sim p_{t}}[u_{t}^{\theta}(x)^{T}u_{t}^{\text{target}}(x)]+\mathbb{E}_{t\sim\text{Unif}_{[0,1]},x\sim p_{t}}[\|u_{t}^{\text{target}}(x) \|^{2}]$$

注意 : 这里最后一项 $\mathbb{E}_{t\sim\text{Unif}_{[0,1]},x\sim p_{t}}[\|u_{t}^{\text{target}}(x) \|^{2}]$ 与训练参数完全无关 , 也就是说在这里可以视其为常数 , 记为$C_1$ , 最后得到 : 

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}]-2 \mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u_{t}^{\text {target}}(x)]+C_{1}$$

再看结果右侧式的中间项 , 目的是将其的 $u^{target}(x)$ 化成 $u^{target}(x|z)$ : 

$$\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{ \theta}(x)^{T}u_{t}^{\text{target}}(x)]=\int\limits_{0}^{1}\int\limits_x p_{t}(x)u_{t}^{\theta}(x)^{T}u_{t}^{ \text{target}}(x) \mathrm{d}x \mathrm{d}t$$

代入u表达式：
$$\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{ \theta}(x)^{T}u_{t}^{\text{target}}(x)]=\int\limits_{0}^{1}\int\limits_x p_{t}( x)u_{t}^{\theta}(x)^{T}\left[\int\limits_z u_{t}^{\text{target}}(x|z)\frac{p_{t}(x|z)p_{ \text{data}}(z)}{p_{t}(x)}\mathrm{d}z\right]\mathrm{d}x \mathrm{d}t$$

调整积分顺序（注意这里得到的是$p_t(x|z)$而不是$p_t(z|x)$，这是用了上一行第一项的$p_t(x)$与积分里面的分母相消而不是对内部用了贝叶斯定理）：
$$\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{ \theta}(x)^{T}u_{t}^{\text{target}}(x)]=\int\limits_{0}^{1}\int\limits_x\int\limits_z u _{t}^{\theta}(x)^{T}u_{t}^{\text{target}}(x|z)p_{t}(x|z)p_{\text{data}}(z) \mathrm{d}z \mathrm{d}x \mathrm{d}t$$

最后发现这个式子可以被视为一个期望 : 首先是 $t$ 服从 $0$ 到 $1$ 均匀分布 , 故 $dt = p(t)dt$ , 最外层 $t$ 积好相当于将t限定为  给定量 , 然后积 $z$ ( 因为积分域为会便利所有变量的全部空间 , 所以前后无所谓 ) , 相当于 $z$ 给定 , $p(x|z)$ 也就成了 $p(x)$ , 最后积分 $x$

$$\mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{ \theta}(x)^{T}u_{t}^{\text{target}}(x)]=\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[u_{t}^{\theta}(x)^{T}u_{t}^{ \text{target}}(x|z)]$$

证完这一步就可以代入回去整合了：
$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}]-2 \mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u_{t}^{\text {target}}(x|z)]+C_{1}$$

然后往回凑完全平方项：

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)\|^{2}]-2 \mathbb{E}_{t\sim\text{Unif},x\sim p_{t}}[u_{t}^{\theta}(x)^{T}u_{t}^{\text {target}}(x|z)]+ \mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_t^{target}(x|z)\|^2] -\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_t^{target}(x|z)\|^2] + C_{1}$$

整理，得到：

$$\mathcal{L}_{\text{FM}}(\theta) {{=}}\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x) - u_t^{target}(x|z)\|^{2}]-\mathbb{E}_{t\sim\text{Unif},z \sim p_{\text{data}},x\sim p_{t}(\cdot|z)}[\|u_t^{target}(x|z)\|^2] + C_{1}$$

而右侧式的第一项为$\mathcal{L}_{\text{CFM}}(\theta)$，第二项为常量（与θ无关），第三项还是常量，进而得证

此时便可以训练了 : 训练得到的 $u^{theta}$ 可以在 Flow Model $dX_t = u(X)dT$ 中对 $X$ 进行模拟而得到 $X_1 \sim p_{data}$ , 这一整个流程也就是 Flow Matching 的全部内容了 .

接下来是一个具体例子 : 已经有了的 Gaussian Probability Path 有
$$u_t^{target}(x|z) = \left ( \dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right )z + \frac{\dot\beta_t}{\beta_t}x$$

就可以根据它得到

$$\mathcal{L}_{\mathrm{CFM}}(\theta) =\mathbb{E}_{t\sim\mathrm{Unif},x\sim \mathcal{N}(\alpha_tz,\beta^2_t,I_d)}[\|u_t^{\theta}(x)-\left ( \dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right )z + \frac{\dot\beta_t}{\beta_t}x\|^2]$$

我们也已提过 $\alpha_t$ 与 $\beta_t$ 的解析式是需要自己设计的 . 甚至可以直接设计为 $\alpha_t = t $ , $\beta = 1 - t$ , 这种情况叫 ( Gaussian ) CondOT Probability Path .式子还能利用此简单些 : $$\mathcal{L}_{\mathrm{CFM}}(\theta) =\mathbb{E}_{t\sim\mathrm{Unif},z \sim p_{data}, \epsilon \sim \mathcal{N}(0,I_d)}[\| u_t^\theta (tz + (1 - t)\epsilon) - (z - \epsilon) \| ^ 2]$$

基于此 , 最后总结 Flow Matching 算法 : 


需求 : 一个数据集 , 一个网络 $u_t^{\theta}$
循环 : 
    1. 抽一个小批量数据 
    2. 从 $[0,1]$ 区间中随机取一个时间 $t$
    3. 并行化地从 $x \sim p_t(\cdot|z)$ 中对 $x$ 取样 , 也就是初始化一个噪音 .
    4. 计算 CFM Loss 的和
    5. 根据 Loss 更新 $\theta$

接下来是 Score Matching : 

首先回忆已经推出来的更新式子
$$\mathrm{d}X_t = \left [u_t^{target}(X_t) + \frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\right ]\mathrm{d}t + \sigma_t\mathrm{d}W_t$$

注意到除了需要训练 $u^{\theta}_t$ 之外我们还要考虑 Score Function . 但实际上对它的处理与$u^{\theta}_t$ 几乎完全一样，可以定义

$$\mathcal{L}_{\text{SM}}(\theta) =\mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}( \cdot|z)}[\|s_{t}^{\theta}(x)-\nabla\log p_{t}(x)\|^{2}]$$
与$$\mathcal{L}_{\text{CSM}}(\theta) =\mathbb{E}_{t\sim\text{Unif},z\sim p_{\text{data}},x\sim p_{t}( \cdot|z)}[\|s_{t}^{\theta}(x)-\nabla\log p_{t}(x|z)\|^{2}]$$
然后用几乎完全一样的证明得到二者差常数量 , 可以训练后者 . 

而最终可以用 $\mathrm{d}X_t = \left [u_t^{\theta}(X_t) + \frac{\sigma_t^2}{2}s^{\theta}_t(X_t)\right ]\mathrm{d}t + \sigma_t\mathrm{d}W_t$ 进行预测 . 

似乎 Score Matching 中训练两组参数会劣于 Flow Matching , 但有这两组数据是可以同时训练 , 甚至在某些情况下互通的 , 因此没有太多额外开销 ( 而为什么 SM 比 FM 麻烦那么多还效果不一定好却仍被提及 , 似乎是因为前者虽然麻烦但是与物理动力学等联系更多 , 反而是更早被发现的 . )

比如在 Gaussian Probability Path 中有 $$u_{t}^{\text{target}}(x|z)=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_ {t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x|z)+\frac{\dot{\alpha}_ {t}}{\alpha_{t}}x$$以及$$u_{t}^{\text{target}}(x)=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_ {t}}-\dot{\beta}_{t}\beta_{t}\right)\nabla\log p_{t}(x)+\frac{\dot{\alpha}_ {t}}{\alpha_{t}}x$$
( 推导涉及的解析式是全的 , 纯粹的代入计算就能得到了 )

也就能得到$$u_{t}^{\theta}(x|z)=\left(\beta_{t}^{2}\frac{\dot{\alpha}_{t}}{\alpha_ {t}}-\dot{\beta}_{t}\beta_{t}\right)s^{\theta}_t+\frac{\dot{\alpha}_ {t}}{\alpha_{t}}x$$

教材的4.3更多是些有关历史上该领域名词的演化与策略的完善优化 , 这里不讨论了 . 

# Part 5 基于流内核的更复杂应用

这部分聚焦到是整体模型构建，重心是两个板块：
1. Conditional Generation ( Guidance ) , 用 Prompt 引导生成图片
2. 神经网络架构 ( 以图片视频为重点分析 )
   
先提第一部分 : 理想情况是从 $p_{data}(x|y)$ 中取样而不是仅靠没有提示词的 $p_{data}(x)$

而这种以提示词 $y$ 为条件的取样 , 将用 Guided 取代 Conditional , 来和 $u_t^{target}(x|z)$ 中的 Conditional 做区分 ( 此处对于 $y$ 只有名字的区别 , 本质没有区别 )

进而得到由 Guided Vector Field $u_t^{\theta}(x|y)$ 组成的 Guided Diffusion Model .

( 对比 $u_t^{\theta}(x|y)$ 与 $u_t^{target}(x)$ 与 $u_t^{target}(x|z)$ 的区别 )

此基础上可以提炼出模型 $\mathrm{d}X_t = u_t^{\theta}(x|y)\mathrm{d}t + \sigma_t\mathrm{d}W_t$ ( 注意这里的 $u$ 是包括进去 Score 信息的 , 不同写法而已 , 毕竟不同文章的操作与描述方法不同 , 不同地方需要的抽象层级也不同 )

而为了能允许语言描述的加入 , 原来仅由图片构建的分布 $p_{data}$ 现在要成为图片与文本组成的联合分布 . 在文本处理上 , 可以一开始就输入标签 , 也可以输入语言再通过其它预训练模型提取信息后作为 $y$ , 这里不具体展开怎么处理了 . 

最终得到一个新的训练目标 

$$\mathcal{L}_{\mathrm{FM}}^{\text{guided}}(\theta) =\mathbb{E}_{t\sim\mathrm{Unif},(z,y)\sim p_{\mathrm{data}}(z,y),x\sim p_{t}(\cdot|z)}[\|u_{t}^{\theta}(x)-u_{t}^{\mathrm{target}}(x)\|^{2}]$$

除此以外理论上似乎就没什么需要额外注意的了 , 但投入训练后发现效果没那么好 .

人们发现人力加强 Guide 的力度应该可以改善 . 问题是该怎么加强 ? 沿着这个思路可以得到一个叫 Classifier-Free Guidance 的方法 .
( 起这个名字是因为原来有种方法涉及到手动设计 Classifier , 但那种方法已经被废弃了 , 所以这里也不多提了 ) .
以下选用 Gaussian Probability Path 进行分析 .

根据之前的分析,用 $x|y$ 替换 $x$ , 有 : 

不妨简记为 $u_{t}^{\text{target}}(x|y)=b_t\nabla\log p_{t}(x)+a_tx$ , 其中有
$$(a_{t},b_{t})=\left(\frac{\dot{\alpha}_{t}}{\alpha_{t}},\frac{\dot{\alpha}_{t }\beta_{t}^{2}-\dot{\beta}_{t}\beta_{t}\alpha_{t}}{\alpha_{t}}\right)$$

接下来对 Score Function 更改结构 , 有 : 
$$\nabla\log p_{t}(x|y)=\nabla\log\left(\frac{p_{t}(x)p_{t}(y|x)}{p_{t}(y)} \right)=\nabla\log p_{t}(x)+\nabla\log p_{t}(y|x)$$
( 因为求梯度是对 $x$ 进行的 , 与 $y$ 无关 , 故 $-\nabla p_t(y) = 0$ )

回代得到 : 

$$u_{t}^{\text{target}}(x|y)=a_{t}x+b_{t}(\nabla\log p_{t}(x)+\nabla\log p_{t}( y|x))=u_{t}^{\text{target}}(x)+b_{t}\nabla\log p_{t}(y|x)$$
( 这里教材 Notes 里比较明显地写反了 )

进而发现可以将 $y$ 脱离出来 , 就可以对后面一项施加权重 , 变成 :
$$\tilde{u}_t(x|y) = u_{t}^{\text{target}}(x)+wb_{t}\nabla\log p_{t}(y|x)$$

可以称 $w$ 为 Guidance Scale ( 一般大于 $1$ ) , 增大其发现确实可以提升表现  .而回代这个新的 $u$ , 可以得到一个比较有意思的式子 :
$$\tilde{u}_{t}(x|y)=(1-w)u_{t}^{\text{target}}(x)+wu_{t}^{\text{target}}(x|y)$$
( 这里的推导过程是普适的 , 对于非 Gaussian Probability Path的情况 , 可以直接引入这个缩放系数 , 让 $\tilde{u}$ 从这里开始 )

也就是说 , 其事实上可以看作 $u^{target}_t(x)$ 与 $u^{target}_t(x|y)$ 两个模型的线性组合 , 但这不代表真的需要两个模型 : 这里涉及 Classifier-Free Guidance 的一个核心技巧 : 令 $u_{t}^{\text{target}}(x) =  u_{t}^{\text{target}}(x|\varnothing)$ . 也就是说 , 之前设计出来数据集被要求返回文本与图片 , 而此方法要求它在每次采样好后算 Loss 前使其文本无效 ( 更改到空标签哪一类 ) , 也就是说有点像 Dropout

对于 Score Matching , 完全一样 , 得到
$$\tilde{s}_{t}(x|y)=(1-w)\nabla\log p_{t}(x|\varnothing)+w\nabla\log p_{t}(x|y)$$

接下来讨论架构 : 

对于一些小模型 , 可以用 MLP 先简单看看效果 , 但显然上限也不会高 . 而大一些的架构核心就是如何压缩与处理信息表示 , 因为原图尺寸是很难接受的 .

第一种常见架构是 U-Net . 它的输入输出形状相同 ( 但可能通道数不同 ) . 包含一系列编码器与解码器 ( 一般是卷积层 , 连带着池化之类的东西 ) , 中间还有处理模块 ( 如果并列着画个图就能得到个 U 槽 ) , 而在编码时增大通道数 , 减小图宽高 , 连接用残差等 . 

第二种是 Diffusion Transformers ( DiTs ) , 其不用卷积而完全用 Transformer , 核心是将图分为小块再对每个小块做 Embedding , 有些借鉴 ViTs

而无论什么架构 , 为了减轻数据量过大的内存开销问题 , 会用 latent diffusion models , 相当于先压缩数据再学习 , 而压缩用的模型也需要提前训练好 , 这里讲的是能否有效压缩与较好地复原

