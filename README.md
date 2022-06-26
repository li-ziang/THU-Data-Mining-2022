# Data-Mining

这是2022年数据挖掘大作业的报告，作业内容为天猫复购概率预测挑战赛项目。项目成员为李伟楷，李子昂，夏箫（以字母排序）。

## 总览

我们实现了三种不同的算法用于商品复购概率，分别为

* XGBoost

  我们使用XGBoost算法，在非图结构数据上进行训练与测试。XGBoost基于Gradient Boosting，使用决策树作为基学习器。Gradient Boosting训练多轮得到多个学习器，每轮的学习器将之前所有学习器输出之和与真实标签的残差作为学习目标，以逐步降低模型在训练集上的偏差，改进学习效果；XGBoost采用同样的思路，通过将损失函数进行二阶Taylor展开，推导每轮的学习目标的表达式，并采用贪心搜索训练决策树。XGBoost在使用决策树时可通过并行化、利用cache等方法加速训练过程。

* MLP

  TODO:

* GraphSAGE

  我们将用户和商家作为异质节点，以商家和用户之间的交易关系作为边，构造异构图，并使用异构图上的神经网络`GraphSAGE`进行学习。

这三个模型分别代表了研究者处理数据挖掘问题的三个阶段。第一个阶段是使用传统的机器学习方法，使用好的特征加上适当的分类器，以及特定的集成学习方法就能得到好的结果。第二阶段是使用深度学习方法，我们使用MLP这一典型的深度学习方法，使用与第一种方法类似的特征工程，并且比较了二者的差异。第三阶段是使用图表示学习方法，这种方法的特征工程与前两种有大的区别，我们特征工程的区别将进行了比较，并给出了相对合适的特征。

## 0. 运行

运行

```python
pip install -r requirments.txt
```

以获取所有依赖。

TODO: 三种模型的运行方式

## 1. 数据分析与预处理

使用data_format1。原始数据文件需要手动放在`./data/taobao/raw`下，具体如下：

```
(base) ziang@ubuntu:/data/ziang/data-mining$ ls data/taobao/raw/
test_format1.csv   user_info_format1.csv
train_format1.csv  user_log_format1.csv
```

DataFrame中有一些数据是缺失的，如果直接将csv读成DataFrame则会出现nan，需要进行判断并将其赋值成合适的值。

异构图：图中包含了424170个独立的用户节点，以及1994个商家节点。在`user_info_format1.csv`中，以用户和商家之间的交易信息为边（同一组用户和商家之间只能有一条边），则一共有14052685条边。如果仅以`train_format1.csv`与`test_format1.csv`中的交易数据建边，则有522341条边，其中train集中有260864条边，test集中有261477条边，他们全部都在前面提到的14052685条边中，这也意味着他们的feature都是可以根据`user_info_format1.csv`中的信息建立的。

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
