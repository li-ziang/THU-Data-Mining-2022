# Data-Mining

这是2022年数据挖掘大作业的报告，作业内容为天猫复购概率预测挑战赛项目。项目成员为李伟楷，李子昂，夏箫（以字母排序）。

## 总览

我们实现了三种不同的算法用于商品复购概率，分别为

* XGBoost

  我们使用XGBoost算法，在非图结构数据上进行训练与测试。XGBoost基于Gradient Boosting，使用决策树作为基学习器。Gradient Boosting训练多轮得到多个学习器，每轮的学习器将之前所有学习器输出之和与真实标签的残差作为学习目标，以逐步降低模型在训练集上的偏差，改进学习效果；XGBoost采用同样的思路，通过将损失函数进行二阶Taylor展开，推导每轮的学习目标的表达式，并采用贪心搜索训练决策树。XGBoost在使用决策树时可通过并行化、利用cache等方法加速训练过程。

* MLP

  MLP是一个经典的神经网络模型。将训练集中的每一个(用户，商家)对作为训练数据，将其统计特征作为输入送入模型，模型最终给出1维的输出。

* GraphSAGE

  我们将用户和商家作为异质节点，以商家和用户之间的交易关系作为边，构造异构图，并使用异构图上的神经网络`GraphSAGE`进行学习。

这三个模型分别代表了研究者处理数据挖掘问题的三个阶段。第一个阶段是使用传统的机器学习方法，使用好的特征加上适当的分类器，以及特定的集成学习方法就能得到好的结果。第二阶段是使用深度学习方法，我们使用MLP这一典型的深度学习方法，使用与第一种方法类似的特征工程，并且比较了二者的差异。第三阶段是使用图表示学习方法，这种方法的特征工程与前两种有大的区别，我们特征工程的区别将进行了比较，并给出了相对合适的特征。

## 0. 运行

运行

```python
pip install -r requirments.txt
```

以获取所有依赖。

运行XGBoost或MLP：python run.py --其他参数

## 1. 数据分析与预处理

使用data_format1。原始数据文件需要手动放在`./data/taobao/raw`下，具体如下：


```
(base) ziang@ubuntu:/data/ziang/data-mining$ ls data/taobao/raw/
test_format1.csv   user_info_format1.csv
train_format1.csv  user_log_format1.csv
```

DataFrame中有一些数据是缺失的，如果直接将csv读成DataFrame则会出现nan，需要进行判断并将其赋值成合适的值。

异构图：图中包含了424170个独立的用户节点，以及1994个商家节点。在`user_info_format1.csv`中，以用户和商家之间的交易信息为边（同一组用户和商家之间只能有一条边），则一共有14052685条边。如果仅以`train_format1.csv`与`test_format1.csv`中的交易数据建边，则有522341条边，其中train集中有260864条边，test集中有261477条边，他们全部都在前面提到的14052685条边中，这也意味着他们的feature都是可以根据`user_info_format1.csv`中的信息建立的。


**一，数据分析：用户信息**

1，训练集大小260864，label=1有15952，占比0.06115

测试集大小261477

共424170个用户

54925330条log记录

 

2，年龄分布：<18岁为1；[18,24]为2； [25,29]为3； [30,34]为4；[35,39]为5；[40,49]为6； > = 50时为7和8; 0表示未知

所有用户的年龄分布：（依次为未知,1,2,3,4,5,6,7,8）

95131

24

52871

111654

79991

40777

35464

6992

1266

​                               

 

训练集里不同年龄段用户的label比例：

age range: 0 ; cnt: 57062 ; #label: 3322 ; label rate: 0.05821737758928884

age range: 1 ; cnt: 13 ; #label: 0 ; label rate: 0.0

age range: 2 ; cnt: 31026 ; #label: 1531 ; label rate: 0.04934571004963579

age range: 3 ; cnt: 69369 ; #label: 4080 ; label rate: 0.058815897591143015

age range: 4 ; cnt: 51235 ; #label: 3444 ; label rate: 0.06721967405094174

age range: 5 ; cnt: 25618 ; #label: 1793 ; label rate: 0.06998985088609572

age range: 6 ; cnt: 21701 ; #label: 1483 ; label rate: 0.0683378646145339

age range: 7 ; cnt: 4120 ; #label: 249 ; label rate: 0.06043689320388349

age range: 8 ; cnt: 720 ; #label: 50 ; label rate: 0.06944444444444445

 

测试集：

age range: 0 ; cnt: 57183

age range: 1 ; cnt: 15

age range: 2 ; cnt: 31359

age range: 3 ; cnt: 69155

age range: 4 ; cnt: 51363

age range: 5 ; cnt: 25614

age range: 6 ; cnt: 21681

age range: 7 ; cnt: 4310

age range: 8 ; cnt: 797

整体上年龄越大复购率越高。训练集和测试集的年龄分布差不多。

 

3，性别分布：0表示女性，1表示男性，2和NULL表示未知

整体用户：

285638女

121670男

16862未知

训练集：

gender: 0 ; cnt: 176414 ; #label: 11387 ; label rate: 0.06454703141474033

gender: 1 ; cnt: 73756 ; #label: 3969 ; label rate: 0.05381257118064971

gender: 2 ; cnt: 10694 ; #label: 596 ; label rate: 0.055732186272676267

 

测试集：

gender: 0 ; cnt: 176277

gender: 1 ; cnt: 74338

gender: 2 ; cnt: 10862

女性的复购率高于男性。测试集和训练集分布比较一致

 

 

**二，数据分析：用户和商家的交互信息**

Label=1：

点击: 14.82541374122367 至少点一次的占比: 0.8952482447342026

加购物车: 0.020937813440320963 至少加一次的占比: 0.014731695085255767

买: 1.6231820461384152 至少买一次的占比: 1.0

收藏: 0.6791624874623872 至少收藏一次的占比: 0.22780842527582748

总log: 17.148696088264796 至少一次记录的占比: 1.0

 

Label=0：

点击: 8.702631965767296 至少点一次的占比: 0.8848484353563729

加购物车: 0.023926961520872803 至少加一次的占比: 0.017900307049062522

买: 1.3206866139674658 至少买一次的占比: 1.0

收藏: 0.3674544326125302 至少收藏一次的占比: 0.18200823152805906

总log: 10.414699973868165 至少一次记录的占比: 1.0


## 2.XGBoost

### 2.1 代码

代码包括`data_xgboost.py`与`run.py`。`data_xgboost.py`为特征工程代码，其中`generate_data`利用data_format1构建特征，输出数据用于训练与测试；如果是首次运行的话，会在源文件所在目录下生成`.cache`目录存放数据处理结果，便于后续使用。`run.py`内使用`xgboost`库中的XGBClassifier，训练模型并输出测试结果。

通过`python run.py --model xgboost`运行代码。

### 2.2 特征工程

使用data_format1，以“用户-商家”对作为样本构建特征，特征围绕用户个体、商家个体以及用户与商家交互信息构建。

用户个体信息含有：

* 用户产生行为的商品数、类别数、商家数、品牌数（4维）
* 用户点击、加入购物车、购买、收藏的次数（4维）
* 用户购买数与点击、加入购物车、收藏次数的比值（3维）
* 用户年龄与性别，用哑变量表示（12维）

商家个体信息含有：

* 商家产生行为的商品数、类别数、商家数、品牌数（4维）
* 商家的所有商品被点击、加入购物车、购买、收藏的次数（4维）
* 商家商品购买数与点击、加入购物车、收藏次数的比值（3维）
* 商家在训练集中拥有的标签为1或为0的买家数（2维）

用户与商家交互信息含有：

* 用户与该商家产生行为的商品数、类别数、品牌数（3维）
* 用户对该商家商品点击、加入购物车、购买、收藏的次数（4维）
* 用户对该商家商品购买数与点击、加入购物车、收藏次数的比值（3维）

样本信息还包括用户与商家的编码（2维），共计48维特征。样本总数为260864。

样本信息处理完成后，将数据集按4:1比例随机划分为训练集与验证集，大小分别为208691$\times$48，52173$\times$48。

### 2.3 效果与分析

模型在验证集上的ROC-AUC可以达到0.717；在比赛平台上ROC-AUC为0.684637，为我们组最终提交成绩。

在上述特征中，“商家在训练集中拥有的标签为1或为0的买家数”这一特征对结果影响最为显著，若不使用这两维特征，模型在验证集上的ROC-AUC会降至0.675。这两个特征由于含义为该商家拥有的重复买家数，可直接反映该商家让用户成为重复买家的能力，因此可以作为预测的重要依据。

此外，我们发现描述“用户与商家交互”的特征，如某用户在某商家处的购买信息，也会对结果造成较大影响。如果去掉此类特征，也即每个样本仅保留对应用户与商家的个体画像后，ROC-AUC下降为0.704，这在一定程度上反映了用户与商家关联信息的重要性。


## 3.MLP

### 3.1代码

run.py中整合了前两种特征工程以及前两种模型（XGBoost和MLP），使用者可以在run.py里自由组合任一种特征工程和模型。run.py里的preprocess1和preprocess2函数为两种不同的特征工程，run_mlp和run_xgb函数为运行两种不同模型的代码。evaluate函数被run_mlp函数调用，对模型在验证集上进行评估。

如果是首次运行代码，代码会先进行数据预处理，读取原始数据路径为data/taobao/raw，预处理结果存在data/taobao/processed1路径下，预处理时间为10分钟左右。预处理生成3个文件：train_x.npy, train_y.npy, test.npy。

如果已经运行过代码了，再次运行代码时会自动到data/taobao/processed1路径下读取预处理结果，不会再次进行预处理。

运行代码可以添加的命令行参数：

```python
parser.add_argument('--hidden_size', type=int, default=32)   # 隐层大小
parser.add_argument('--lr', type=float, default=2e-3)        # lr
parser.add_argument('--wd', type=float, default=0)           # wd
parser.add_argument('--num_epoch', type=int, default=100)    # 训练轮数
parser.add_argument('--batch_size', type=int, default=1000)  # batch大小
parser.add_argument('--dropout', type=float, default=0.1)    # dropout
parser.add_argument('--num_layers', type=int, default=3)     # 层数
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'mlp'])  # 模型
parser.add_argument('--feature', type=int, default=1, choices=[1, 2], help='which feature to use')     # 使用第一种还是第二种特征工程
parser.add_argument('--new_preprocess', action='store_true', help='whether to run a new preprocess')   # 添加这个参数会重新预处理，更换特征工程时使用
parser.add_argument('--no_onehot', action='store_true', help='whether to use onehot representation of age and gender')    # 添加这个参数会使得预处理不使用onehot表示，仅限XGBoost模型
```



### 3.2特征工程

第二种特征工程仅以“用户-商家”pair为基本单元进行数据统计，没有像第一种特征工程一样单独对用户个人的信息或商家个人的数据做统计。

缺失值处理：
用户性别的缺失值替换成1种额外性别（这样就共有3种性别），年龄的缺失值替换成1种额外年龄（这样就共有9种年龄），商品brand的缺失值替换为-1。

使用41维特征：
用户性别(one-hot，3维)、年龄(one-hot，8维)，
点击数，加购物车数，买的数，收藏数，log数，点击数占log数比例，加购物车数占log数比例，购买数占比，收藏数占比（共9维），
浏览的天数，有浏览的时候平均每天浏览数、点赞数、加购物车数、购买数、收藏数（共6维），
浏览的商品的总数、点赞商品总数、加购物车商品总数、购买总数、收藏总数（5维），
浏览的品牌的总数、点赞品牌总数、加购物车品牌总数、购买总数、收藏总数（5维），
浏览的商品类别总数，点赞类别总数、加购物车类别总数、购买类别总数、收藏总数（共5维）



### 3.3模型训练机制

使用3层MLP模型。按8:2比例划分训练集和验证集。采用带早停机制的训练，当验证集上的rocauc低于此前最高验证集rocauc时把patience加一，当patience达到总轮数一半时停止训练。然后用验证集上跑出最高rocauc时刻的模型去跑测试集。



### 3.4前两种特征工程与算法的分析

#### 模型与特征工程对比

我们对两钟特征工程在两种模型上的表现进行了对比实验。我们按照8:2的比例划分训练集和验证集，下表给出了验证集上的结果，下表结果为5次结果的平均：

|         | 特征工程1 | 特征工程2 |
| ------- | --------- | --------- |
| XGBoost | todo      | 0.6223    |
| MLP     | 0.5000    | 0.6183    |

第一种特征工程在XGBoost上表现很好但在MLP上表现很差。第二种特征工程在两个模型上表现差不多，都不算很好。

从模型比较的角度来说，XGBoost明显好于MLP。这在深度学习时代比较反常，因为模型表现并不


## 4. GraphSAGE

### 4.1 代码

模型主要由两部分文件构成，`dataset.py`中的`TaobaoDataset`是一个[PyG](https://www.pyg.org/)风格的图数据集，在这个文件中我进行了原始数据的处理，并将其处理成一个可以直接用torch加载的`.pt`文件。原始文件需要手动放在`./data/taobao/raw`下，生成的文件将会储存在`./data/taobao/processed`下，具体如下:

```shell
(base) ziang@ubuntu:/data/ziang/data-mining$ ls data/taobao/raw/
test_format1.csv   user_info_format1.csv
train_format1.csv  user_log_format1.csv
(base) ziang@ubuntu:/data/ziang/data-mining$ ls data/taobao/processed/
data.pt  pre_filter.pt  pre_transform.pt
```

在第一次运行`TaobaoDataset`时，会从`raw`中读取文件并运行`process`函数，最后生成`data.pt`文件。之后如果再次运行`TaobaoDataset`，则会从`data.pt`中直接读取处理好的文件。需要运行训练和评估时，直接运行`python heterogeneous.py`即可。

### 4.2特征工程

我尝试直接从`user_log_format1.csv`建立图。

在异构图学习中，节点与边都可以给特征，我设计的基础特征如下：

* 用户节点（冒号后为对应的维度）

年龄：[1]，性别：[1]。

张量大小：[424170,2]

* 商家节点

使用一个商家是否卖过某个category的商品，也就是判断是否包含了某个`cat_id`是否出现在了当前商家的销售信息中，如果出现了则对应的值设置为1，否则为0。特征的维度为[1658]，也就是独立的`cat_id`的个数。

张量大小：[1994, 1672]

* 边

我直接使用了`user_log_format1.csv`中的数据，举个例子：

```python
user_log.head()
Out[158]: 
  user_id  item_id  cat_id  seller_id  brand_id  time_stamp  action_type
  0   328862   323294     833       2882    2661.0         829            0
```

可以看到`user_log`的存在形式是`user_id->seller_id`，在基础的版本中，边的是没有feature的。但是边的feature其实是易得的，这里可以直接使用`item_id  cat_id  seller_id time_stamp  action_type`作为对应的`user_id ->brand_id`二元组的feature。

张量大小：[2, 5551143]\(用于描述稀疏邻接矩阵)，[5551143,5]\(边对应的feature，基础版本中没有使用这个，下面的讨论默认是只使用了前面的稀疏邻接矩阵)。

**Mark**：从输入上看，仍然还有`user_log`中的交易信息没有存放到图中，特征工程的重点应该放在如何将这五个信息加入到图中。在这之后，如果有一个合理的图学习框架，则可以有效的学到每个节点相应的表示。TODO:要改。

### 4.3 基于显存的特征工程优化

(1) 对于图结构数据的深度学习，比较重要的一个因素就是显存问题，这是因为图数据的节点和边的feature是一次性输入的，没有办法像cv或nlp中的数据分batch进行输入，所以必须严格控制显存。可以看到，三个组feature占用的显存大小大致为`M(边)=10*M(商家)=100*M(用户)`，这是因为边的总数较多，而且用户节点的feature维度较少造成的。考虑到数量上`N(商家)<<N(用户)<<N(边)`，**合理的做法应该是优先减少边的个数，其次是加大商家的feature，最后才是加大用户的feature。**在后面的优化中，我也会尊崇这个法则。

(2) 其次，在实验上，使用1.2中的所有边的稀疏邻接矩阵需要33G的显存才能将实验顺利运行完成（这里我使用的是一张48G的Titan 8000），远远超出了一般显卡的内存空间。而如果我们在构造异构图的时候只使用`train`和`test`中存在的边（一共有522341条，见第1部分），需要的显存将会大幅减小至10G左右，是一般显卡可以承受的。两种feature构造方式对结果的影响见4.4。

​	在4.5的Table 1中，我发现只使用`train`和`test`中存在的边的结果比使用完整的边的稀疏邻接矩阵邻接矩阵效果更好。

(3) 对于商家节点，因为其个数较少，并且往往一个商家节点会与很多用户节点相连，（$\frac{d(商家)}{d(用户)}=213$其中d表示相应节点的度），所以在商家的节点中提供更多的信息其实是显存友好的。从图结构上考虑，商家节点有更多的可能性将其feature传递到周围的用户节点，这样使用少的显存储存和传递了更多的信息。4.2中的商家节点只使用了`cat_id`作为feature，其实可以增加的feature是`brand_id`。

​	在4.5的Table 3中，我发现增加`brand_id`后在valid上的结果有所提升。

(4) 对于用户节点，可以增加的feature是购买过的`cat_id`和`brand_id`，由于`cat_id`和`brand_id`的数量分别为1658和8844，直接使用这个长度（1658+8844）的特征向量会需要大约25G的显存，24G的显卡装不下。我的解决方案是使用在总记录中出现频率最高的128个`cat_id`和512个`brand_id`作为feature，这相当于是进行了一次特征提取，在减少显存的同时增加了突出了主要特征，优化后的feature维度为$128+512=640$。

​	在4.5的Table 2中，我发现了增加`cat_id`和`brand_id`后模型在valid上的结果有所提升，同时，在使用了出现频率最高的feature后，模型需要的显存降低到了9G，并且在valid上的结果有所提升。

### 4.4 方法

模型的结构如下：

![image-20220626073723494](/Users/liziang/Library/Application Support/typora-user-images/image-20220626073723494.png)

左图是一个常见的同质图的图深度学习模型，右图是其模型结构对应的异构图学习模型，输入为`x_user`和`x_seller`，输出为`out_user`与`out_seller`，它们都是对应的特征向量。两图的差别在于右图会有两个pipeline分别处理不同类型的节点，同时如果两个不同类型的两个的节点之间也会进行数据的传递。详细过程可以在右图中看出。

其中`x_seller`就是直接由4.2中描述的feature进行输入，而`x_user`的输入分为三部分，其中age和gender都需要通过一个embedding layer，而`cat_id`和`brand_id`则可以直接拼起来作为输入。embedding layer的维度对结果有比较大的影响，详见Table 4。

![image-20220626072819552](./image-20220626072819552.png)

模型输出部分将user embedding，seller  embedding和edge feature拼在一起，过一个线性层，输出结果。

![image-20220626074403788](/Users/liziang/Library/Application Support/typora-user-images/image-20220626074403788.png)

### 4.5 实验

**Table 1**

我比较了四种方法的实验结果，它们分别为1) 只使用`train`和`test`中存在的边，2)使用`train`和`test`中存在的边+sample 0.1其他边

3)使用`train`和`test`中存在的边+sample 0.5其他边，4)所有边

结果如下

| Model Number        | 1     | 2     | 3     | 4     |
| ------------------- | ----- | ----- | ----- | ----- |
| Valid roc_auc_score | 0.571 | 0.560 | 0.527 | 0.501 |

**Table 2**

| Model               | model+seller(`brand_id`+`cat_id`) | model+top seller(`brand_id`+`cat_id`) |
| ------------------- | --------------------------------- | ------------------------------------- |
| Valid roc_auc_score | 0.575                             | 0.595                                 |

**Table 3**

| Model               | model+top seller(`brand_id`+`cat_id`) | model+top seller(`brand_id`+`cat_id`)+user(`brand_id`) |
| ------------------- | ------------------------------------- | ------------------------------------------------------ |
| Valid roc_auc_score | 0.595                                 | **0.609**                                              |

**Table 4**

dropout=0

| Model(age emb dim+ gender emb dim) | 24+8  | 12+4  | 8+4   |
| ---------------------------------- | ----- | ----- | ----- |
| Valid roc_auc_score                | 0.596 | 0.601 | 0.596 |

**Table 5**

同时我尝试了不同的emb dim与dropout的组合，发现dropout=0时效果最好。

dropout=0.5

| Model(age emb dim+ gender emb dim) | 8+4   | 32+4  | 64+16 |
| ---------------------------------- | ----- | ----- | ----- |
| Valid roc_auc_score                | 0.551 | 0.548 | 0.548 |

## 5 分工

我们采用了每个人主要负责一个模型，其他同学辅助的形式。我们在特征工程上彼此分享了对自己模型有显著贡献的特征。`随机森林+XGBoost`的主要负责人是夏箫，`MLP`的主要负责人是李伟楷，`GraphSAGE`的主要负责人是李子昂。此外，李子昂和李伟楷负责了数据的统计分析和预处理。总体上三个人贡献相同。
