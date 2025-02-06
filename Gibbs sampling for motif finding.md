﻿# Gibbs sampling for motif finding

实验流程与分析 | by Xingyue WANG

---
 [TOC]

### 一、算法思路
在统计学中，Gibbs采样是一种马尔可夫链蒙特卡罗 （Markov chain Monte Carlo，MCMC）算法，当直接抽样比较困难时，它可以从指定的多元概率分布中近似得到一系列观测结果。
马尔可夫链是指当前状态只与前一状态相关，而Gibbs采样中，每次随机采样得到数据就相当于马尔可夫链中不同的状态。在寻找moitf时，每次采样选取的一条链上的各个位置的得分只与上一状态得到的打分矩阵相关，直至状态收敛，也就是得到的矩阵收敛。

> * 主要步骤：
> *  第一步：在18条序列上随机截取8bp长度的短序列构建位置频数矩阵（PFM）
> * 第二步：将PFM转变为打分矩阵：将PFM转化为PPM（位置概率矩阵），再除以背景值，得到打分矩阵
> * 第三步：从PFM的18条短序列中随机扔去一条，提取该短序列对应的序列全长，分别对这条序列的两个方向及其互补序列的两个方向进行切片打分（该算法的前提是转录因子的结合不具有方向性，因此相当于对一条链一共进行了四次打分），首先取出每次打分最高的序列，再比较得到的四条得分最高的短序列，取出其中得分最高的放回PFM
> * 第四步：依次循环第二步和第三步999次
> * 第五步：将最后一次得到的PFM转变为PPM和打分矩阵
> * 第六步：利用最后得到的打分矩阵对前两条序列（cole1、ecoarabop）进行打分，并根据得分绘制柱状图

### 二、编程及注释
在本实验中motif finding由Python实现，绘制sequence logo由R实现。

------------------
#### **一、程序基本组成**
本实验中共编写了三个函数，应用于1000次sampling：

1. `score_matrix()`：
输入：序列全长、打分矩阵
输出：得分最高的8nt短序列及其得分、整个序列的不同位置的得分列表（相隔1nt)
作用：对序列进行切片打分，得到得分最高的8nt短序列
2. `build_pwm()`：
输入：PFM
输出：打分矩阵，位置频率矩阵（PPM)
作用：将位置频数矩阵（PFM）转换成打分矩阵
3. `count_score()`：
输入：打分矩阵
输出：新的PFM，整个序列的位置得分列表
作用：打分，并将得分最高的8nt放入PFM构建新的PFM
本函数中调用了函数`score_matrix()`

----------
#### **二、变量的储存形式**

 1. 矩阵（二维数组）
本实验中所有矩阵均采用二维数组进行储存，通过数组元素的索引来进行数据的提取，用于之后的计算。在这部分使用了Python的第三方库：numpy。
 2. 序列（列表）
本实验中18条长序列和用于构建PFM的18条8nt的短序列都是采用列表形式存储的，便于在之后随机采样过程中对应8nt的短序列和原序列。
 3. 背景值的计算（字典）
背景值计算时需要统计四种碱基在整个矩阵中出现的次数，因此我将碱基作为键，该种碱基出现次数作为值，以字典的形式储存。
 4. 截取的motif及其得分（字典、列表）
在对长序列以1为步长进行切片打分时，切下来的8nt短序列和其得分采用字典储存，键为序列，值为该序列对应的得分。采用字典储存还有一个优点：便于寻找最高的得分及其对应的motif。
由于之后需要利用得分绘制柱状图，因此我将得分单独存为一个列表，列表的索引可以作为横坐标（表征该段短序列在原长序列上的位置），纵坐标就是得分。
 5. 用于绘制sequence logo的短序列（TXT文件）
每次sampling得到的18条长度为8nt的短序列都是以列表形式储存的，但是由于Python中没有用于绘制sequence logo的第三方库，因此需要将第1000次得到的短序列列表输出。考虑到用于绘制sequence logo的R包ggseqlogo需要的数据输入格式，我用读写文件的形式，将列表元素（8nt短序列）以换行符分隔，写入新的文件"seq_for_logo.txt"中，之后在R中可以直接读取、画图。

---------------
#### **三、打分矩阵的构建**
在本实验中为了之后的打分方便，我直接将PFM转化为打分矩阵进行打分。
具体公式如下：
$$
M_{i,j}=\frac{n_{i,j}/18}{x_i/(18*8)}=\frac{n_{i,j}*8}{x_i}
$$

$$
M_{i,j}：打分矩阵中碱基i在第j位的打分值
$$

$$
n_{i,j}：在原PFM中，碱基i在第j位出现的次数
$$

$$
x_i：在原PFM中，碱基i在整个矩阵中出现的次数
$$




--------------
#### **四、完整程序流程**
##### **1、构建matrix及绘制柱状图（Python）**

```python
#第一个PFM是截取每条链随机位置的连续8bp
#在sampling过程中也是每次随机选取一条链
#由于设置的是随机种子seed(),所以每次结果都不相同

#用到的第三方库
#numpy和Biopython用于构建矩阵和矩阵相关运算
#random用于生成随机数
#matplotlib用于绘制柱状图
import numpy  
from Bio import motifs
from Bio.Seq import Seq
import random
import matplotlib.pyplot as plt

random.seed()  #设置随机种子，每次结果不相同

#循环读取，并以列表形式储存FASTA文件内容
#motif_crp.fa中存储18条fasta格式序列
#seq_for_logo.txt用于储存绘制sequence logo所需要的8nt短序列
fo=open("motif_crp.fa","r")  
fs=open("seq_for_logo.txt","w")
l=fo.read()
li=l.split("\n\n")
li_seq=[]  
for i in li[:-1]:
  li_seq.append(i.replace("\n",":")[1:])


#切片打分，得出最大得分的短seq(8bp)
def score_matrix(sequence,matrix):
    start=0
    lenth=len(sequence)
    end=lenth-7
    seq_score_dic={}  #利用字典记录该条序列上的每8个碱基对应的得分
    seq_score_list=[] #利用列表储存得到序列打分，用于之后绘制柱状图
    for point in range(start,end):
        score=1
        p=0
        m_y=0  #代表碱基在8nt中的位置，用于之后的打分
        seq=sequence[point:point+8]
        #将四种碱基转换成四个数字，有利于之后的打分
        for each in seq:
            if each=="A":
                m_x = 0
            elif each=="T":
                m_x = 1
            elif each=="G":
                m_x = 2
            elif each=="C":
                m_x = 3
            p=matrix[m_x][m_y]
            score=score*p
            m_y+=1
        seq_score_dic[seq]=round(score,6)
        seq_score_list.append(round(score,6))
    max_seq=max(seq_score_dic,key=seq_score_dic.get)  #返回得分最高的序列
    max_seq_score=seq_score_dic[max_seq]
    return (max_seq,max_seq_score,seq_score_list)

#将PFM转换成打分矩阵
#构建出的矩阵一列代表一个位点，从左至右分别为0，1，2，3，4，5，6，7
#矩阵的每一横行代表一种碱基，从上到下分别为A,T,G,C
def build_pwm(seq_ma):
    m=motifs.create(seq_ma)
    dNTP=["A","T","G","C"]
    dNTP_copy=[1,2,3,4]
    site_li=["0","1","2","3","4","5","6","7"]
    
    #统计矩阵中四种碱基的个数，用于背景值计算
    su=0
    dntp_dic={}  #储存每一种碱基在整个矩阵中的总个数
    for dntp in dNTP:
        su=0
        dntp_list=m.counts[dntp]
        for x in dntp_list:
            su=su+x 
        dntp_dic[dntp]=su

    #每一个位点四种碱基的数量为18
    #将PFM转换为打分矩阵
    matrix=numpy.zeros(shape=(4,8))  #打分矩阵
    pwm=numpy.zeros(shape=(4,8))    #位置频率矩阵
    for dntp_c in dNTP_copy:
        lis=[]
        lis_pwm=[]
        if dntp_c==1:
            dntp="A"
        elif dntp_c==2:
            dntp="T"
        elif dntp_c==3:
            dntp="G"
        elif dntp_c==4:
            dntp="C"
        for y in site_li:
            y=eval(y)
            c_dntp=m.counts[dntp,y]
            b_dntp=dntp_dic[dntp]
            new=round(c_dntp*8/b_dntp,6) #保留小数点后6为小数
            new_pwm=round(c_dntp/18,6)
            lis.append(new)
            lis_pwm.append(new_pwm)
        matrix[dntp_c-1]=lis
        pwm[dntp_c-1]=lis_pwm
    return (matrix,pwm)

#打分并将得分最高的8nt放入PFM构建新的PFM
#由于转录因子结合没有方向性，所以将对原序列和其互补序列分别从两个方向进行打分(四次），从中选取得分最高的短序列
def count_score(matrix):
    s=random.randint(0,17) #随机选择18条序列中的一个
    
    #提取序列（原序列），对该序列进行打分，调用score_matrix()函数
    seq_for_count=li_seq[s].split(":")[1]  
    (max_seq,max_seq_score,seq_score_list)=score_matrix(seq_for_count,matrix)  

    #将原序列反向计算得分，同样提取最高得分的8nt
    seq_for_count_re=seq_for_count[::-1]
    (max_seq_re,max_seq_re_score,seq_re_score_list)=score_matrix(seq_for_count_re,matrix)

    #计算反向互补链的得分，使用Python第三方库Biopython
    seq_co_re=str(Seq(seq_for_count).reverse_complement())
    (max_seq_co_re,max_seq_co_re_score,seq_co_re_score_list)=score_matrix(seq_co_re,matrix)

    #计算互补链的得分
    seq_co=str(Seq(seq_for_count).complement())
    (max_seq_co,max_seq_co_score,seq_co_score_list)=score_matrix(seq_co,matrix)

    #比对找到的四个8nt短seq，找出其中得分最大的那条
    seq_score_max_dic={}
    seq_score_max_dic[max_seq]=max_seq_score
    seq_score_max_dic[max_seq_re]=max_seq_re_score
    seq_score_max_dic[max_seq_co_re]=max_seq_co_re_score
    seq_score_max_dic[max_seq_co]=max_seq_co_score
    seq_score_max=max(seq_score_max_dic,key=seq_score_max_dic.get)
    seq_score_max_list=seq_score_max.split(":")
    seq_score_max=seq_score_max_list[0]
    
    #用新的得分最高的8nt短序列替换原来的短序列
    seq_ma[s]=seq_score_max

    return (seq_ma,seq_score_list)

#以上为自定义函数
#以下为函数主体

#随机提取8nt序列，创建第一个矩阵
seq_ma=[]
i=0
for seq in li_seq:
    o=random.randint(0,98)
    s=seq.split(":")
    seq_ma.append(s[1][o:o+8])

#循环体
#构建PWM、打分、找出最大值的那条链并用新链替换旧链、构建新的PFM
count=0
while count<=999:
    (matrix,pwm)=build_pwm(seq_ma)
    (seq_ma,seq_score_list)=count_score(matrix)
    count+=1
   
#结果输出 
(matrix_final,pwm_final)=build_pwm(seq_ma)       
print(pwm_final)  #输出最后的PPM
print(matrix_final)
logo_plot = "\n".join(seq_ma) + "\n"
fs.write(logo_plot)  #将最后得到的18条短seq存入文件，用于生成sequence logo

#循环针对前两条序列cole1和ecoarabop进行四个方向的打分并绘制柱形图
seq_hist=[]
seq1=li_seq[0].split(":")[1]
seq2=li_seq[1].split(":")[1]
seq_hist.append(seq1)
seq_hist.append(seq2)
seq_name={}
seq_name[seq[0]]="cole1"
seq_name[seq[1]]="ecoarabop"

for seq1 in seq_hist:
    #原序列正向打分
    (max_seq1,max_seq1_score,seq1_score_list)=score_matrix(seq1,matrix_final)
    #原序列反向打分
    seq1_re=seq1[::-1]
    (max_seq1_re,max_seq1_re_score,seq1_re_score_list)=score_matrix(seq1_re,matrix_final)
    #反向互补序列打分
    seq1_co_re=str(Seq(seq1).reverse_complement())
    (max_seq1_co_re,max_seq1_co_re_score,seq1_co_re_score_list)=score_matrix(seq1_co_re,matrix)
    #互补序列打分
    seq1_co=str(Seq(seq1).complement())
    (max_seq1_co,max_seq1_co_score,seq1_co_score_list)=score_matrix(seq1_co,matrix)

    x=numpy.array(range(1,99))  #横坐标，位置
    y1=numpy.array(seq1_score_list)  #纵坐标，正向得分
    y1_re=numpy.array(seq1_re_score_list)[::-1]  #反向得分
    y1_co_re=numpy.array(seq1_co_re_score_list)[::-1]  #反向互补得分
    y1_co=numpy.array(seq1_co_score_list)  #互补得分
    plt.bar(x,y1,color='green', label='ori seq')
    plt.bar(x,y1_re,color='red', label='rev seq')
    plt.bar(x,y1_co_re,color='blue', label='rev_com seq')
    plt.bar(x,y1_co,color='yellow', label='com seq')
    plt.legend()
    plt.title(seq_name[seq[0]])
    plt.xlabel("site")
    plt.ylabel
    plt.show()

fo.close()  #关闭文件
fs.close()
```
-------------
##### **2、根据得到的matrix绘制seq logo（R）**
由于Python没有用于绘制sequence logo的库，所以该部分使用R语言的`ggseqlogo`包

```
#将最后得到的18条8nt的短seq存到seq_for_logo.txt中，再读取绘制sequence logo
seq_logo=readLines("seq_for_logo.txt")
ggseqlogo(seq_logo)
```
### 三、实验结果

#### 一、结果图示
由于每次实验结果都不相同，我只选取三次随机实验的结果进行展示（为了实验的可重复性，我设置了给定种子）： 

**1. 结果一：seed(10)**

> * 矩阵（位置频率矩阵）：
>   $$
>   \begin{array}{c|lcr}
>    & \text{1} & \text{2} & \text{3} & \text{4}& \text{5}& \text{6}& \text{7}& \text{8}\\
>   \hline
>   A & 0 & 0 & 0 & 0.944444 & 0.611111 & 0.6111111 & 0.277778 & 0.166667
>    \\
>   T & 0.944444 & 0.333333 & 0 & 0.055556 & 0.055556 & 0 & 0.333333 & 0.833333
>    \\
>   G & 0.055556 & 0.666667 & 0.166667 & 0 & 0 & 0.166667 & 0 & 0
>    \\
>   C & 0 & 0 & 0.833333 & 0 & 0.333333 & 0.222222 & 0.388889 & 0
>   \end{array}
>   $$
>   
>
> * 该矩阵对应的sequence logo：
> ![sequence logo][1]

---------------
> * 第一条序列cole1打分的柱状图：
**ori seq**：原序列的从左到右的打分情况（假设为5'到3'端）
**rev seq**：原序列从右到左的打分情况（3'到5'端）
**rev_com seq**：反向互补序列的打分情况（从3'到5'端）
**com seq**：互补序列的打分情况（从5'到3'端）
由图可以看出得分最高的motif位于原序列的反方向上
![cole1的得分柱状图][2]


> * 第二条序列ecoarabop打分的柱状图：
由图可以看出得分最高的motif也是位于原序列的反方向上
![ecoarabop的得分柱状图][3]

 **2. 结果二：seed(77)**
 > * 矩阵（位置频率矩阵）：
 >  $$
 >  \begin{array}{c|lcr}
 >   & \text{1} & \text{2} & \text{3} & \text{4}& \text{5}& \text{6}& \text{7}& \text{8}\\
 >  \hline
 >  A & 0 & 0.777778 & 0.333333 & 0 & 0 & 0.222222 & 0.722222 & 0.111111
 >   \\
 >  T & 0 & 0 & 0 & 0.833333 & 0.944444 & 0.666667 & 0.166667 & 0.055556
 >   \\
 >  G & 0.833333 & 0.222222 & 0.111111 & 0.111111 & 0.055556 & 0 & 0.111111 & 0.111111
 >   \\
 >  C & 0.166667 & 0 & 0.555556 & 0.055556 & 0 & 0.111111 & 0 & 0.722222
 >  \end{array}
 >  $$
 >  
 >  
 > * 矩阵对应的sequence logo：
 > ![sequence logo][4]

---------------
> * 第一条序列cole1打分的柱状图：
由图可以看出得分最高的motif位于原序列的反方向上
![cole1的得分柱状图][5]
> * 第二条序列ecoarabop打分的柱状图：
由图可以看出得分最高的motif位于互补链上
![ecoarabop的得分柱状图][6]

**3. 结果三：seed(1649)**

 > * 矩阵（位置频率矩阵）：
 >  $$
 >  \begin{array}{c|lcr}
 >   & \text{1} & \text{2} & \text{3} & \text{4}& \text{5}& \text{6}& \text{7}& \text{8}\\
 >  \hline
 >  A & 0.833333 & 0 & 0.666667 & 0 & 0 & 0 & 0 & 0.722222
 >   \\
 >  T & 0.166667 & 0.5 & 0 & 0 & 0.388889 & 0.444444 & 0.888889 & 0.277778
 >   \\
 >  G & 0 & 0.5 & 0.333333 & 0 & 0.611111 & 0.388889 & 0.111111 & 0
 >   \\
 >  C & 0 & 0 & 0 & 1 & 0 & 0.166667 & 0 & 0
 >  \end{array}
 >  $$
 >  
 >  
 > * 矩阵对应的sequence logo：
 > ![sequence logo][7]

---------------
> * 第一条序列cole1两个方向打分的柱状图：
由图可以看出得分最高的motif位于互补链上
![cole1的得分柱状图][8]
> * 第二条序列ecoarabop两个方向打分的柱状图：
由图可以看出得分最高的motif位于原序列上
![ecoarabop的得分柱状图][9]

--------------------------------------------
#### 二、结果分析
**1. 每次运行结果不同**

+ 虽然以上三种结果都不相同，但是它的结果都收敛。
+ 每次结果都不同的现象可以反映上一个矩阵对本次打分的影响，即马尔可夫链中上一状态对当前状态的影响。
+ 一旦矩阵中某一位点的某种碱基的打分为0，那么在该位点就不可能出现这种碱基，也就使得该位点是该碱基的短序列不可能用于更新矩阵，从而高得分8nt短序列范围（用于矩阵更新）逐渐缩小，矩阵很快收敛。
+ 由于前一矩阵对打分的影响很大，因此每次矩阵的更新都会对下一次的矩阵产生很大影响。即使是第一个随机选取的矩阵，也会对之后的收敛矩阵的形成产生影响。且每次实验都是随机选取第一个矩阵和随机选择序列进行打分，因此导致每次的运行结果都不相同。

**2. motif的位置**


+ 在本实验中，我对一条序列进行了4次打分，分别为：原序列的两个方向及其互补序列的两个方向。使用该算法的前提是：转录因子在序列上的结合不具有方向性，且在正负链中都可以结合。
+ 依据：[The orientation of transcription factor binding site motifs in gene promoter regions: does it matter?](http://ghosertblog.github.com) 
+ 这篇16年发表在BMC Genomics上的文章通过统计学方法研究拟南芥中293个已经过文献报道的非回文转录因子结合位点序列和10个核心启动子基序，得出结论：转录因子的结合不具有方向性，转录因子是否发挥功能的关键在于其是否结合在TFBS上，而不是其方向。转录因子也可以结合在互补序列上，尤其是当存在转录因子二聚体时。
+ 在以上算法前提下，由最后的柱状图显示的得分情况可以看到，最后找到的最高分的motif的位置有的位于原序列上，有的位于互补链上，符合算法的设计思路。

**3.Gibbs采样算法缺点**

+ 结果的确定性不高：每次运行结果都不同；且变换motif的长度，得到的结果的差别也非常大。

