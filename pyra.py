#!/usr/bin/env python3
#_∗_coding: utf-8 _∗_


"""
"""
import os
import io
import string
import re
import sys
import datetime
import copy
import json
import operator #数学计算操作符
import random
import time
import matplotlib.pyplot as plt
import pylab as mpl     #import matplotlib as mpl
#import networkx as nx
#import pydot
#from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch_geometric
from torch_geometric.data import InMemoryDataset
import torch_geometric.data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv,MLP,GINConv,GraphConv,SAGPooling
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
#from torch_geometric.data import HeteroData

import pickle
import networkx as nx
import pydot
import tqdm



    

#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体 FangSong,SimHei
#mpl.rcParams['font.serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14


sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.append("D:\\TexasHoldem\\LuaDemo\\agent") 
sys.path.append("D:\\TexasHoldem\\TexasAI") 
from Ccardev.evaluatordll import *
print('sys.path=',sys.path)

linestyles = [#
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dashed', 'dashed'),    # Same as '--' or (0, (5, 5)))
     ('dashdot', 'dashdot'),
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('loosely dotted',        (0, (1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
colors_use=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
    '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00', '#1f77b4', '#800080',
    '#959595', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3',
    '#8595e1', '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593',
    '#c6dec7', '#ead3c6', '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7',
    '#f3e1eb', '#f6c4e1', '#f79cd4']
markline=[
'.'       ,#point marker
','       ,#pixel marker
'o'       ,#circle marker
'v'       ,#triangle_down marker
'^'       ,#triangle_up marker
'<'       ,#triangle_left marker
'>'       ,#triangle_right marker
'1'       ,#tri_down marker
'2'       ,#tri_up marker
'3'       ,#tri_left marker
'4'       ,#tri_right marker
's'       ,#square marker
'p'       ,#pentagon marker
'*'       ,#star marker
'h'       ,#hexagon1 marker
'H'       ,#hexagon2 marker
'+'       ,#plus marker
'x'       ,#x marker
'D'       ,#diamond marker
'd'       ,#thin_diamond marker
'|'       ,#vline marker
'_'        #hline marker
]



#创建保存数据集便于后面复用
class GameHisTreeDataset(InMemoryDataset):

    ftout=False
    
    def __init__(self, root, transform=None, pre_transform=None, ftout=False):
        super().__init__(root, transform, pre_transform)

        self.ftout=ftout
        #root可以看做是数据集的名，也是数据存放的根目录，下面会有默认的raw_dir和processed_dir
        #每一个数据集都需要指定一个根目录，根目录下面需要分为两个文件夹，
        #一个是raw_dir，这个表示下载的原始数据的存放位置，
        #另一个是processed_dir，表示处理后的数据集存放位置。

        #若存在数据则会自动从root目录下的数据目录下自动加载
        self.data, self.slices = torch.load(self.processed_paths[0])

        #如果要考虑给出更多的数据集信息可以进一步处理给出
        #需要结合后面定义的带property修饰的函数给出
        #dataset对象有很多属性的处理可以用类似的方法得到
        #这些属性可以用dir查看
        #注意这里由于要访问Dataset内部的Data数据，这并不推荐，所以正常情况下会报警告
        #可以使用_data来代替data来消除这个警告
        #self.numofgraphfeatures=self._data.x.size()[-1]


    #返回num_features属性
    '''
    @property
    def num_features(self) ->int:
        return self.numofgraphfeatures
    '''

    #返回数据集源文件名
    @property
    def raw_file_names(self):
        #因为是本地创建，不用下载解压，所有没有必要修改
        return ['some_file_1']
    

    #用于从网上下载数据集
    def download(self):
        # Download to `self.raw_dir`.
        #因为不下载，所以直接pass
        #download_url(url, self.raw_dir)
        pass


    #返回process方法所需的保存文件名。用于保存数据的文件名
    @property
    def processed_file_names(self):
        return ['GHTdataNUM.pt']
    

    
    #当不下载，则生成数据集所用的方法
    def process(self):

        # Read data into huge `Data` list.
        data_list = []

        if self.root=="GHTdata3000":
            data_list=preparedata(3000,self.ftout)
        elif self.root=="GHTdata1000":
            data_list=preparedata(1000,self.ftout)
        elif self.root=="GHTdata750":
            data_list=preparedata(750,self.ftout)
        elif self.root=="GHTdata500":
            data_list=preparedata(500,self.ftout)
        elif self.root=="GHTdata250":
            data_list=preparedata(250,self.ftout)
        else:
            ngsamples=int(self.root[7:])
            data_list=preparedata(ngsamples,self.ftout)

        #处理过滤器
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        #处理数据转换
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #先把数据整合再存储
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#显示图信息
def graph_info(data):
    '''
    args:
         data: torch_geometric.data.Data
    '''
    # Gather some statistics about the first graph.
    print('G=',data)
    print(f'Number of nodes: {data.x.size()[0]}')
    print(f'Number of edges: {data.edge_index.size()[1]}')
    print(f'Average node degree: {data.edge_index.size()[1] / data.x.size()[0]:.2f}')
    #print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    #print(f'Is undirected: {data.is_undirected()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Class of graph: {data.y}')
    print('x[:3] ft  of G:',data.x[:3])
    print('edgeindex of G:',data.edge_index)
    
    return None


#测试数据集建立和读取
def testDATAsetcreate(ngsamples=3000,ftout=False):

    dataset=GameHisTreeDataset(f"GHTdata{ngsamples}",ftout=ftout)
    #print(dir(dataset))
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    flg_compare=False
    if flg_compare:
        datasetcp=originaldata(ngsamples)

    if type(ftout)==type(False) :#只有全特征的时候才画图看看，因为画图依赖于一些特征的信息
        for i in range(len(dataset)):
            if i%(len(dataset)/14) in [0]:
                print(f"\n=====================\n{i=:}")
                G=dataset[i]
                graph_info(G)
                drawfeatureactiontree(G)
                #plt.show()
                #print('press anykey to continue:')
                #anykey=input()

            if flg_compare:
                gdict1,gsb=datasetcp[i]
                drawactiontree(gdict1,gsb)

    return None



# ACPCserver平台给出的结果文件的处理
datausers=[]
recAllresults=[]
recAllagentsname=[]
#第三个参数flagsinglefile是一个标记处理单文件的标记，若为true，收集的数据仅是当前给定文件的
#若非flase那么是集合多个文件的。
def logfiledealing(filename,myname,flagsinglefile=True,bigblindfirst=False,flagfoldseen=True):
    global datausers
    global recAllresults
    global recAllagentsname
    
    if flagsinglefile:
        recAllresults=[]

    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    for r in resdata:
        lino+=1
        if (r.count("STATE")>0):
            data=ACPClogmsgTodata(r.strip(),myname,bigblindfirst,flagfoldseen)
            #print('data=',data)
            recAllresults.append(data)

    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in recAllresults[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    datausers=recAllresults
    datalength=len(recAllresults)
    return datalength




# 将acpc的log的信息转换为data数据
# 注意这是两人情况下用的，因为其中很多处理都是针对两人做的。
# 输入是log文件中的字符串和玩家的名字
# BigBlindfirst选项用于控制大盲注先行的情况，从acpc收集的数据看是有大盲注线性的。
# 而默认是小盲注先的。
params_sb = 50
params_bb = 100
params_stack = 20000
params_maxr = 3  #一轮最大raise次数
def ACPClogmsgTodata(msg,myname,BigBlindfirst=False,flagfoldseen=True):

    data={} #字典

    #print('in msg=',msg)
    m1=re.search("^STATE:(\d*):([^:]*):(.*):(.*):(.*)",msg.strip())
    #print('m1=',m1)
    
    hand_id=int(m1.group(1))
    actions=m1.group(2).strip()
    cards=m1.group(3)
    if BigBlindfirst:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')
        win_money.reverse()
        playernames.reverse()
        winmoneyadjusted="|".join(win_money)
        playernamesadjusted="|".join(playernames)
    else:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')

    msgsbfirst="STATE:"+str(hand_id)+":"+actions+":"

    room_number=2
    position=playernames.index(myname) #我方的位置
    #print('position=',position)
    name=myname
    opposition=1-position
    opp_post=1-position        #对手位置
    opname=playernames[opposition]
    

    m2=re.search("([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",actions)
    #print('m2=',m2,m2.group(0))
    preflop_actions = m2.group(1)
    flop_actions = m2.group(2)
    turn_actions = m2.group(3)
    river_actions = m2.group(4)
    lastaction=''
    if actions:
        lastaction=actions[-1]
    if preflop_actions=='':
        flagFirstAction=True
    else:
        flagFirstAction=False

    #print("cards",cards)
    m3 = re.search("([^\|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",cards.strip())
    #print('m3=',m3,m3.group(0))
    if BigBlindfirst:
        hand_p1=m3.group(2)
        hand_p2=m3.group(1)
    else:
        hand_p1=m3.group(1)
        hand_p2=m3.group(2)
    
    flopcds=m3.group(3)
    turncds=m3.group(4)
    rivercds=m3.group(5)

    if BigBlindfirst:
        msgsbfirst+=hand_p1+"|"+hand_p2+"/"+"/".join([flopcds,turncds,rivercds])+":"+winmoneyadjusted+":"+playernamesadjusted
    #print("flopcds=",flopcds)
    #print("turncds=",turncds)
    #print("rivercds=",rivercds)

    #位置和局序号
    data['hand_id']=hand_id

    #手牌
    if position==0 :
        data['private_card']=[hand_p1[:2],hand_p1[2:]]
    else:
        data['private_card']=[hand_p2[:2],hand_p2[2:]]

    #print('hand_p1=',hand_p1)
    #print('hand_p2=',hand_p2)
    if hand_p1 and hand_p2 :
        data['player_card']=[[hand_p1[:2],hand_p1[2:]],[hand_p2[:2],hand_p2[2:]]]
    else:
        if position==0 :
            data['player_card']=[[hand_p1[:2],hand_p1[2:]],[]]
        else:
            data['player_card']=[[],[hand_p2[:2],hand_p2[2:]]]
    
    ophand=data['player_card'][opposition]
    if ophand:
        ophandidx=HandtoIdx(cardint[ophand[0]],cardint[ophand[1]])
        data['ophandidx']=ophandidx

    if not flagfoldseen and lastaction=='f': #当对手手牌数据不可观测时，根据动作f来去掉
        data['player_card'][opposition]=[]


    #公共牌
    street=1
    actionstrings=[preflop_actions]
    data['public_card']=[]
    if flopcds :
        data['public_card']=[flopcds[:2],flopcds[2:4],flopcds[4:]]
        street=2
        actionstrings.append(flop_actions)
    if turncds :
        data['public_card'].append(turncds)
        street=3
        actionstrings.append(turn_actions)
    if rivercds :
        data['public_card'].append(rivercds)
        street=4
        actionstrings.append(river_actions)
    data['street']=street
    #print('actionstrings=',actionstrings)

    #动作和下注额
    #要特别注意：下面的代码是很多地方都是基于两人对局做了特殊处理的
    #A raise is valid if a) it raises by at least one chip, b) the player has sufficient money in their stack to
    #raise to the value, and c) the raise would put the player all-in (they have spent all their chips) or the
    #amount they are raising by is at least as large as the big blind and the raise-by amount for any other
    #raise in the round. 
    #注意acpc协议中r****实际是raiseto，就是全局加注到多少(而不是一个轮次的)
    #一次raise实际下注额是call上的+对手raise增加的额度+大盲注
    #那么raiseto的最小值为：原来已经下注的额度+前一次raise增加的额度+大盲注
    money_beted_now=[params_sb,params_bb] #两个人对局：位置0的下注额，位置1的下注额
    money_beted_rnd=[0,0,0,0] #四个轮次 
    actions=[]
    actions_all=[]
    tmp_lastAction=[] #最后一个动作，倒数第一个动作
    tmp_lastSecAction=[] #最后第二个动作，倒数第二个动作
    tmp_lastThiAction=[] #倒数第三个动作
    tmp_roundactstr=''   #描述当前轮已经做的动作
    tmp_roundactnum=0    #描述当前要做的动作是第几个动作
    tmp_op_tideway=''    #描述前面各轮对手做的最强动作，主要是c和r
    tmp_my_tideway=''    #描述前面各轮自己做的最强动作
    raisetimes=0
    raisemprev=0
    raise_amount=0
    actpos=0    #各动作的所对应的玩家的位置
    for i  in range(len(actionstrings)):
        #tmp_lastRoundAction='' #当前轮的动作记录
        #tmp_lastMyRoundAction='' #当前轮的动作记录

        actions_remainder=actionstrings[i]
        actions_round=[]
        raisetimes=0 #一轮中raise的次数，各轮需要分别统计
        raisemprev=0 #一轮中一次raise增加的下注额度，各轮需要分别统计
        actsn=0      #一轮中动作的序号
        actstr=''    #一轮中的动作历史记录：用每个动作的首字母表示的
        acttide=''
        actmytide=''
        while actions_remainder != '':
            actsn+=1 #每一轮第一个动作序号为1，第二个动作序号为2，后续为3,4,5...
            if i==0: #preflop轮小盲先行
                actpos=1-actsn%2
            else: #flop/turn/river轮大盲先行
                actpos=actsn%2
            #print('actsn=',actsn)
            #print('actpos=',actpos)
            parsed_chunk = ''
            if actions_remainder.startswith("c"):
                #call 改成check
                if money_beted_now[actpos]==money_beted_now[1-actpos]:
                    actionstr="check"
                else:
                    actionstr="call"
                actions_round.append({"action":actionstr,"position": actpos,'mynowbet':money_beted_now[1-actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                actions_all.append({"action":actionstr,"position": actpos,'mynowbet':money_beted_now[1-actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                parsed_chunk = "c"
                if opp_post==actpos:
                    acttide='c'
                else:
                    actmytide='c'
                money_beted_now[actpos]=money_beted_now[1-actpos]
            elif actions_remainder.startswith("r"):
                #print('actions_remainder=',actions_remainder)
                raise_amount = int(re.search("^r(\d*).*",actions_remainder).group(1))
                bet_now=raise_amount
                parsed_chunk = "r" + str(raise_amount)
                #print('raise_amount=',raise_amount)
                if opp_post==actpos:
                    acttide='r'
                else:
                    actmytide='r'
                raisetimes+=1
                #raise的额度就是在call平基础上增加的额度
                #raisemprev=raise_amount-max(money_beted_now)
                #raise_amount就是raise后的下注额，即raiseto的额度
                #注意acpc协议是全局的raiseto，而cisia则是当前轮的raiseto
                #acpc
                #actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                #actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                #parsed_chunk = "r" + str(raise_amount)
                #cisia
                if i==0:
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                else:
                    raise_amount=raise_amount-money_beted_rnd[i-1]
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                money_beted_now[actpos]=bet_now
            elif actions_remainder.startswith("f"):
                actions_round.append({"action":"fold","position": actpos,'mynowbet':money_beted_now[actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                actions_all.append({"action":"fold","position": actpos,'mynowbet':money_beted_now[actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                parsed_chunk = "f"
            else:
                print("wrong action string")
            actstr+=parsed_chunk[0]
            tmp_lastThiAction=tmp_lastSecAction
            tmp_lastSecAction=tmp_lastAction
            tmp_lastAction=[actions_round[-1]["action"],actpos,i+1,actions_round[-1]] #记录三个参数：动作名，行动者，轮次
            #tmp_lastMyRoundAction=tmp_lastRoundAction
            #tmp_lastRoundAction=actions_round[-1]["action"] #当前轮的动作记录
            
            actions_remainder = actions_remainder.replace(parsed_chunk,"",1)

        if money_beted_now[0]==money_beted_now[1]:
            money_beted_rnd[i]=money_beted_now[0]
        else:
            #print("not equal bet in round:",i)
            pass

        actions.append(actions_round)
        if acttide != '':
            tmp_op_tideway+=acttide
        if actmytide != '':
            tmp_my_tideway+=actmytide
        tmp_roundactstr=actstr
        tmp_roundactnum=actsn+1
    
    data['action_history']=actions
    data['act_next_sn']=tmp_roundactnum
    data['act_his_rnd']=tmp_roundactstr
    data['act_op_tide']=tmp_op_tideway
    tmp_mytd_plus=''
    for x in tmp_my_tideway:
        if x=='r':
            tmp_mytd_plus+='+'
        else:
            tmp_mytd_plus+='-'
    data['act_my_tide']=tmp_mytd_plus


    #判断决策点信息
    #这里虽然可能轮次变化后最后一个动作可能是前一轮的，但很巧的是
    #下面的逻辑也没有问题，因为前一个动作若是r的画，那么轮次必定没有结束
    #所以面临的决策点是pcl是没有问题的。
    if position==0 :
        flagDecisionPt=1
    else:
        flagDecisionPt=0

    if tmp_lastAction:
        if tmp_lastAction[0][0]=="r":
            flagDecisionPt=1 #pcl point
        else:
            flagDecisionPt=0 #pck point
    
    #记录对手的动作类型
    flag_opaction=None
    
    if tmp_lastAction:
        if tmp_lastAction[1]==position:#当最后一个动作是自己的时就一定在轮次交界处
            #我方的动作必然是c或者f
            #因此对手的动作必然是call或者raise
            #若最后一个动作在第一轮，则对手必然是在pcl
            if street==2: # cc|flop，rc|flop
                flagOpDecisionPt=1 #pcl point,前一次对手面对的决策点类型
            else: # 比如：  rrc|turn
                if tmp_lastThiAction:
                    if tmp_lastThiAction[0][0]=='r':
                        flagOpDecisionPt=1
                    else: # 比如：crc|turn
                        flagOpDecisionPt=0 #pck point,前一次对手面对的决策点类型
                else:
                    flagOpDecisionPt=1
        else:
            if tmp_lastSecAction:
                if tmp_lastSecAction[0][0]=='r':
                    flagOpDecisionPt=1 
                else:
                    flagOpDecisionPt=0

                #对手不是轮次开始时的动作
                #注意这种情况：MATCHSTATE:0:5:cc/:KhTh|/8dAs8s，下，已经进入到下一轮，但对手的动作是上一轮的，这时我方不决策，所以判断不判断不影响
                if tmp_lastAction[0][0]=='c':
                    flag_opaction='call'
                else:
                    invest=tmp_lastAction[-1]['mynowbet']-tmp_lastAction[-1]['myprebet']
                    prepot=tmp_lastAction[-1]['myprebet']+tmp_lastAction[-1]['opprebet']
                    #print('prepot=',prepot,'raise_amount=',invest)
                    if invest < prepot:
                        flag_opaction='raise0'
                    elif invest < 2*prepot:
                        flag_opaction='raise1'
                    else:
                        flag_opaction='raise2'

            else:# 若倒数第二个动作不存在，且不是我做的倒数第一个动作，必然是对手做的，那么就是每局开始的时候
                flagOpDecisionPt=1

                #对手在轮次开始时的动作
                if tmp_lastAction[0][0]=='c':
                    if street==1:
                        flag_opaction='call'
                    else:
                        flag_opaction='check'
                else:
                    invest=tmp_lastAction[-1]['mynowbet']-tmp_lastAction[-1]['myprebet']
                    prepot=tmp_lastAction[-1]['myprebet']+tmp_lastAction[-1]['opprebet']
                    #print('prepot=',prepot,'raise_amount=',invest)
                    if invest < prepot:
                        flag_opaction='bet0'
                    elif invest < 2*prepot:
                        flag_opaction='bet1'
                    else:
                        flag_opaction='bet2'

    else:#没有动作则无需考虑
        flagOpDecisionPt=-1

    data['LastAction']=tmp_lastAction
    data['LastSecAction']=tmp_lastSecAction
    data['LastTrdAction']=tmp_lastThiAction
    data['flagFirstAction']=flagFirstAction
    data['flagDecisionPt']=flagDecisionPt
    data['flagOpDecisionPt']=flagOpDecisionPt
    data['flagOpActtype']=flag_opaction

    #玩家信息
    data['players']=[]
    for i in range(room_number):
        data['players'].append({"position":i,"money_bet":money_beted_now[i],"money_left":params_stack-money_beted_now[i],'total_money':params_stack})
    data['roundbet']=money_beted_rnd
    data['pots']=money_beted_now

    #legal_action
    legalact=[]
    #当前最小的加注额应等于大盲注+对手的加注额度(对手在平基础上增加的额度)
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]不相等时，必然还在一个轮次内
    # 那么对手的加注额必然是：abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]相等时，要么新一轮开始或者一轮前面对手c
    # 那么对手的加注额为0也等于abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 所以最小加注额如下是对的：
    raisetorangemin=abs(money_beted_now[actpos]-money_beted_now[1-actpos])+params_bb
    
    if money_beted_now[actpos]==money_beted_now[1-actpos]:#这里是典型的基于2人的考虑的处理
        legalact.append("check")
        #通常相等的时候raise是没有多次的
        legalact.append("raise")
    else:
        legalact.append("call")
        legalact.append("fold")
        if raisetimes<params_maxr:
            legalact.append("raise")
    data["legal_actions"]=legalact
    if "raise" in legalact:
        raisetorangemin+=max(money_beted_now)
        if raisetorangemin>params_stack:
            raisetorangemin=params_stack

    #注意acpc的服务器的raise的额度可以自动调整的，默认是raiseto，若这个值过小，则会自动调整调整到call后加上该值 
    #所以提示的额度最好使用加注至(即raiseto)，分两种，一种是而且是全局的"raise_to_range"，
    #另一种是像自动化所那样使用一轮的raiseto，使用"raise_range"。
    if "raise" in legalact:#raise的范围是一轮中的值
        if street ==1:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto
            data["raise_range"]=[raisetorangemin,params_stack]
        else:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto。因为从第二轮开始street=2，要减去第一轮的下注，第一轮在money_beted_rnd列表中位置是0。
            data["raise_range"]=[raisetorangemin-money_beted_rnd[street-2],params_stack-money_beted_rnd[street-2]]
    
    #info,action_position
    data['info']='state'
    if lastaction=='f' or (hand_p1 and hand_p2) :
        actpos=2 #为了给出action_position=-1，所以设置为2
        data["info"]="result"
    data['position']=position

    #print("actions_all=",actions_all,len(actions_all))
    #print("actions=",actions,len(actions))
    #print("street=",street)
    #if street==2: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==3: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==4: print("str=",street,actions[street-1],len(actions[street-1]))
    if len(actions_all)==0:#当全局没有任何动作时是小盲
        data["action_position"]=0
    elif street==2 and len(actions[1])==0:#当flop没有任何动作时是大盲
        data["action_position"]=1
    elif street==3 and len(actions[2])==0:#当turn没有任何动作时是大盲
        data["action_position"]=1
    elif street==4 and len(actions[3])==0:#当river全局没有任何动作时是大盲
        data["action_position"]=1
    else:#其它情况下，当前需要动作的玩家是最后一个动作的玩家的对手
        data["action_position"]=1-actpos 


    #一局的结果可以直接算出来
    '''
    win_money=[0,0]
    if hand_p1 and hand_p2:
        rk1=gethandrank(data['player_card'][0],data['public_card'])
        rk2=gethandrank(data['player_card'][1],data['public_card'])
        if rk1>rk2:
            win_money=[money_beted_now[1],-money_beted_now[1]]
        elif rk1==rk2:
            win_money=[0,0]
        else:
            win_money=[-money_beted_now[0],money_beted_now[0]]
    if lastaction=='f':
        win_money=[0,0]
        foldpos=actions_all[-1]["position"]
        win_money[foldpos]=-money_beted_now[foldpos]
        win_money[1-foldpos]=money_beted_now[foldpos]
    '''

    if data["info"]=="result":
        for i in range(room_number):
            data['players'][i]["win_money"]=win_money[i]
    
    data['players'][position]["name"]=name
    data['players'][1-position]["name"]=opname
    data['msgorig']=msg+':BBfirst:'+str(BigBlindfirst)
    if BigBlindfirst:
        data['msgsbfirst']=msgsbfirst

    return data






#从所有玩家的log文件准备数据
#图数据+对手标签
def preparedata(ngamesamples,ftout=False):
    #filenames=[]
    #filenames.append(filename)
    #outfilename='Match-{}.{}-all.log'.format(playername,opname)

    '''
    datasetgh=[]
    playername='ASHE'
    opname='ElDonatoro'
    logfiledealing(f'Match-{playername}.{opname}-all.log',playername+'_2pn_2017',bigblindfirst=True)
    actiontreegen(0,3000)
    gsb=drawactiontree(GameTreeRecPoszero,myposSB=True)  #我方小盲注位时的博弈图
    datasetgh.append(gsb)
    gbb=drawactiontree(GameTreeRecPosones,myposSB=False) #我方大盲注位时的博弈图
    datasetgh.append(gbb)
    '''
    
    playername='ASHE'
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    
    #完全的数据准备
    nsamples=ngamesamples #3000   #500局一个统计数据
    datasetgh=[]
    for opname in opnames:
        filename=f"Match-{playername}.{opname}-all.log"
        ndata=logfiledealing(filename,playername+'_2pn_2017',bigblindfirst=True)
        for k in range(int(ndata/nsamples)):
            actiontreegen(k*nsamples,(k+1)*nsamples)
            gsb=featureactiontree(GameTreeRecPoszero,myposSB=True,tagetlabel=opnames.index(opname))  #我方小盲注位时的博弈图
            gbb=featureactiontree(GameTreeRecPosones,myposSB=False,tagetlabel=opnames.index(opname)) #我方大盲注位时的博弈图
            
            #将两个位置的图合并成一个图
            gsbandbb = torch_geometric.data.Data()
            gsbandbb.x=torch.cat([gsb.x,gbb.x],0)

            if  (type(ftout) != type(False)) and (ftout in list(range(7))):
                ftleft=list(range(7))
                ftleft.remove(ftout)
                xftleft=gsbandbb.x[:,ftleft]
                gsbandbb.x=xftleft

            nsbnode=len(gsb.x)
            edgeidxbb=gbb.edge_index+nsbnode
            gsbandbb.edge_index=torch.cat([gsb.edge_index,edgeidxbb],1)
            gsbandbb.y=gsb.y
            datasetgh.append(gsbandbb)

        #print('anykey to conitnue')
        #anykey=input()
    
    return datasetgh





#模拟在线博弈时获得的数据，从第10局开始，每隔10局更新一个图，直到500局
#用于观察识别器对于类型的判断的概率输出。
#每一类对手的都生成一个数据集，并保存到文件中
#从所有玩家的log文件准备数据
#图数据+对手标签
def preparedataonline(ngstt=6,ngend=400,nginter=6,ftout=False):
    
    playername='ASHE'
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    
    #完全的数据准备
    for opname in opnames:
        datasetgh=[]
        filename=f"onlinedata\\{playername}.{opname}.27.1.log"
        ndata=logfiledealing(filename,playername+'_2pn_2017',bigblindfirst=True)
        for k in range(ngstt,ngend,nginter): #nginter局一个图
            actiontreegen(0,k)
            gsb=featureactiontree(GameTreeRecPoszero,myposSB=True,tagetlabel=opnames.index(opname))  #我方小盲注位时的博弈图
            gbb=featureactiontree(GameTreeRecPosones,myposSB=False,tagetlabel=opnames.index(opname)) #我方大盲注位时的博弈图
            
            #将两个位置的图合并成一个图
            gsbandbb = torch_geometric.data.Data()
            gsbandbb.x=torch.cat([gsb.x,gbb.x],0)

            if  (type(ftout) != type(False)) and (ftout in list(range(7))):
                ftleft=list(range(7))
                ftleft.remove(ftout)
                xftleft=gsbandbb.x[:,ftleft]
                gsbandbb.x=xftleft

            nsbnode=len(gsb.x)
            edgeidxbb=gbb.edge_index+nsbnode
            gsbandbb.edge_index=torch.cat([gsb.edge_index,edgeidxbb],1)
            gsbandbb.y=gsb.y
            datasetgh.append(gsbandbb)
        torch.save(datasetgh,f"GHT-online-{playername}-{opname}.pt")
        #print('anykey to conitnue')
        #anykey=input()
    
    return None







#从所有玩家的log文件准备数据
#原始的图数据
def originaldata(ngamesamples):

    playername='ASHE'
    opnames=['ElDonatoro' #,'Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                #'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    
    #完全的数据准备
    nsamples=ngamesamples #3000   #500局一个统计数据
    datasetgh=[]
    for opname in opnames:
        filename=f"Match-{playername}.{opname}-all.log"
        ndata=logfiledealing(filename,playername+'_2pn_2017',bigblindfirst=True)
        for k in range(int(ndata/nsamples)):
            actiontreegen(k*nsamples,(k+1)*nsamples)
            datasetgh.append([copy.deepcopy(GameTreeRecPoszero),True])
            datasetgh.append([copy.deepcopy(GameTreeRecPosones),False])

        #print('anykey to conitnue')
        #anykey=input()
    
    f = open(f'Dataset-graph-original-{ngamesamples}.dat', 'wb')
    pickle.dump(datasetgh, f, protocol=4)
    f.close()
    
    return datasetgh




#根据每局建立动作历史树
flg_round=[False,False,False,False]
node_round=[None,None,None,None]

#我方位于大盲注，而对手位于小盲注时的博弈历史树
def actionhistreeones(data,flag_output=False):
    global GameTreeRecPosones
    global flg_round,node_round

    #print('data=',data)

    if data['position']==0:
        pass
    else:
        if 'node0' not in GameTreeRecPosones:
            GameTreeRecPosones['node0']={'id':'node0','actname':'1','actseq':'1','access':1,
            'games':[data['hand_id']],'layer':10,
            'parent':None,'grandpa':None,'parlst':[],'child':[],'actpos':None}
            #注意：'actseq'是最后一次到当前节点时的路径的字符串，而不是所有可能到达路径的字符串
        else:
            GameTreeRecPosones['node0']['access']+=1
            GameTreeRecPosones['node0']['games'].append(data['hand_id'])

        currentnode='node0'

        #'action_history': [[{'action': 'r365', 'position': 0, 'mynowbet': 365, 'myprebet': 50, 'opprebet': 100}, 
        #{'action': 'call', 'position': 1, 'mynowbet': 365, 'myprebet': 100, 'opprebet': 365}], 
        #[{'action': 'check', 'position': 1, 'mynowbet': 365, 'myprebet': 365, 'opprebet': 365}, 
        #{'action': 'r885', 'position': 0, 'mynowbet': 1250, 'myprebet': 365, 'opprebet': 365}, 
        #{'action': 'fold', 'position': 1, 'mynowbet': 365, 'myprebet': 365, 'opprebet': 1250}]]


        round=1
        for acthislst in data['action_history']:
            actlayer=0
            for actdict in acthislst:
                if actdict:
                    actlayer+=1
                    if flag_output: print('actdict',actdict)
                    actname=actdict['action'][0]
                    if (actname=='c' or actname=='r') and actdict['mynowbet']>=20000:
                        actname='a'
                    actpos=actdict['position']

                    flag_child_exist=False
                    #若当前节点存在子节点时
                    if GameTreeRecPosones[currentnode]['child']:
                        actnamechilds=[GameTreeRecPosones[id]['actname'] for id in GameTreeRecPosones[currentnode]['child']]
                        if actname in actnamechilds:
                            for id in GameTreeRecPosones[currentnode]['child']:
                                if GameTreeRecPosones[id]['actname']==actname:
                                    GameTreeRecPosones[id]['access']+=1
                                    GameTreeRecPosones[id]['layer']=round*10+actlayer
                                    GameTreeRecPosones[id]['games'].append(data['hand_id'])
                                    GameTreeRecPosones[id]['parent']=currentnode
                                    GameTreeRecPosones[id]['grandpa']=GameTreeRecPosones[currentnode]['parent']
                                    if currentnode not in GameTreeRecPosones[id]['parlst']:
                                        GameTreeRecPosones[id]['parlst'].append(currentnode)

                                    if GameTreeRecPosones[currentnode]['actname'] in ['2','3','4']:
                                        grandpa=GameTreeRecPosones[id]['grandpa']
                                        GameTreeRecPosones[id]['actseq']=GameTreeRecPosones[grandpa]['actseq']+GameTreeRecPosones[currentnode]['actname']+actname
                                    else:
                                        GameTreeRecPosones[id]['actseq']=GameTreeRecPosones[currentnode]['actseq']+actname

                                    currentnode=id
                                    flag_child_exist=True
                                    if flag_output: print('child found',currentnode,GameTreeRecPosones[currentnode])
                                    break

                    if not flag_child_exist:#若当前节点不存在子节点时，创建新节点
                        newnodeid='node'+str(len(GameTreeRecPosones))
                        newnodeactname=actname
                        newnodeaccess=1
                        newnodelayer=round*10+actlayer
                        newnodeparent=currentnode
                        newnodegrandpa=GameTreeRecPosones[currentnode]['parent']
                        newnodeparlst=[currentnode]
                        newnodechild=[]

                        newnodeactseq=''
                        if GameTreeRecPosones[currentnode]['actname'] in ['2','3','4']:
                            #print('parent',currentnode,GameTreeRecPosones[currentnode])
                            #print('grandpa',newnodegrandpa,GameTreeRecPosones[newnodegrandpa])
                            newnodeactseq=GameTreeRecPosones[newnodegrandpa]['actseq']+GameTreeRecPosones[currentnode]['actname']+actname
                        else:
                            newnodeactseq=GameTreeRecPosones[currentnode]['actseq']+actname

                        GameTreeRecPosones[newnodeid]={'id':newnodeid,'actname':newnodeactname,
                        'access':newnodeaccess,'games':[data['hand_id']],'parent':newnodeparent,
                        'grandpa':newnodegrandpa,'child':newnodechild,'actseq':newnodeactseq,
                        'layer':newnodelayer,
                        'parlst':newnodeparlst,
                        'actpos':actpos}
                        GameTreeRecPosones[currentnode]['child'].append(newnodeid)
                        if flag_output: print('child build1',newnodeid,GameTreeRecPosones[newnodeid])
                        currentnode=newnodeid

                if flag_output:
                    print('press any key to continue:')
                    anykey=input()

            #一轮结束后
            round+=1
            if round>4 or GameTreeRecPosones[currentnode]['actname']=='f' or actdict['mynowbet']>=20000:
                pass
            else:
                actname=str(round)

                flg_round_exist=False
                if GameTreeRecPosones[currentnode]['child']:
                    actnamechilds=[GameTreeRecPosones[id]['actname'] for id in GameTreeRecPosones[currentnode]['child']]
                    if actname in actnamechilds:
                        for id in GameTreeRecPosones[currentnode]['child']:
                            if GameTreeRecPosones[id]['actname']==actname:
                                GameTreeRecPosones[id]['access']+=1
                                GameTreeRecPosones[id]['layer']=round*10
                                GameTreeRecPosones[id]['games'].append(data['hand_id'])
                                GameTreeRecPosones[id]['parent']=currentnode
                                GameTreeRecPosones[id]['grandpa']=GameTreeRecPosones[currentnode]['parent']
                                if currentnode not in GameTreeRecPosones[id]['parlst']:
                                    GameTreeRecPosones[id]['parlst'].append(currentnode)
                                currentnode=id
                                flg_round_exist=True
                                if flag_output: print('child found-round',currentnode,GameTreeRecPosones[currentnode])
                                break
                if not flg_round_exist:
                    #一种方式式完全的新增节点
                    #newnodeid='node'+str(len(GameTreeRecPosones))
                    #另一种方式是判断后用已有的节点
                    if flg_round[round-1]:
                        id=node_round[round-1]
                        GameTreeRecPosones[currentnode]['child'].append(id)
                        GameTreeRecPosones[id]['layer']=round*10
                        GameTreeRecPosones[id]['access']+=1
                        GameTreeRecPosones[id]['games'].append(data['hand_id'])
                        GameTreeRecPosones[id]['parent']=currentnode
                        GameTreeRecPosones[id]['grandpa']=GameTreeRecPosones[currentnode]['parent']
                        if currentnode not in GameTreeRecPosones[id]['parlst']:
                            GameTreeRecPosones[id]['parlst'].append(currentnode)
                        currentnode=id
                        if flag_output: print('child found-round-saved',currentnode,GameTreeRecPosones[currentnode])
                    else:
                        newnodeid='node'+str(len(GameTreeRecPosones))
                        newnodeactname=actname
                        newnodeaccess=1
                        newnodelayer=round*10
                        newnodeparent=currentnode
                        newnodegrandpa=GameTreeRecPosones[currentnode]['parent']
                        newnodeparlst=[currentnode]
                        newnodechild=[]
                        
                        GameTreeRecPosones[newnodeid]={'id':newnodeid,'actname':newnodeactname,
                        'access':newnodeaccess,'games':[data['hand_id']],
                        'grandpa':newnodegrandpa,'parlst':newnodeparlst,
                        'layer':newnodelayer,
                        'parent':newnodeparent,'child':newnodechild}
                        flg_round[round-1]=True
                        node_round[round-1]=newnodeid
                        print('*********flg_round=',flg_round)
                        print('*********node_round=',node_round)
                        GameTreeRecPosones[currentnode]['child'].append(newnodeid)
                        if flag_output: print('child build1-round',newnodeid,GameTreeRecPosones[newnodeid])
                        currentnode=newnodeid

            if flag_output:
                print('press any key to continue:')
                anykey=input()


        #print('update GameTreeRecPosones=',GameTreeRecPosones)

    return None


#我方位于小盲注，而对手位于大盲注时的博弈历史树
def actionhistreezero(data,flag_output=False):
    global GameTreeRecPoszero
    global flg_round,node_round

    #print('data=',data)

    if data['position']==1:
        pass
    else:
        if 'node0' not in GameTreeRecPoszero:
            GameTreeRecPoszero['node0']={'id':'node0','actname':'1','actseq':'1','access':1,
            'games':[data['hand_id']],'layer':10,
            'parent':None,'grandpa':None,'parlst':[],'child':[],'actpos':None}
            #注意：'actseq'是最后一次到当前节点时的路径的字符串，而不是所有可能到达路径的字符串
        else:
            GameTreeRecPoszero['node0']['access']+=1
            GameTreeRecPoszero['node0']['games'].append(data['hand_id'])

        currentnode='node0'

        #'action_history': [[{'action': 'r365', 'position': 0, 'mynowbet': 365, 'myprebet': 50, 'opprebet': 100}, 
        #{'action': 'call', 'position': 1, 'mynowbet': 365, 'myprebet': 100, 'opprebet': 365}], 
        #[{'action': 'check', 'position': 1, 'mynowbet': 365, 'myprebet': 365, 'opprebet': 365}, 
        #{'action': 'r885', 'position': 0, 'mynowbet': 1250, 'myprebet': 365, 'opprebet': 365}, 
        #{'action': 'fold', 'position': 1, 'mynowbet': 365, 'myprebet': 365, 'opprebet': 1250}]]


        round=1
        for acthislst in data['action_history']:
            actlayer=0
            for actdict in acthislst:
                if actdict:
                    actlayer+=1
                    if flag_output: print('actdict',actdict)
                    actname=actdict['action'][0]
                    if (actname=='c' or actname=='r') and actdict['mynowbet']>=20000:
                        actname='a'
                    actpos=actdict['position']

                    flag_child_exist=False
                    #若当前节点存在子节点时
                    if GameTreeRecPoszero[currentnode]['child']:
                        actnamechilds=[GameTreeRecPoszero[id]['actname'] for id in GameTreeRecPoszero[currentnode]['child']]
                        if actname in actnamechilds:
                            for id in GameTreeRecPoszero[currentnode]['child']:
                                if GameTreeRecPoszero[id]['actname']==actname:
                                    GameTreeRecPoszero[id]['access']+=1
                                    GameTreeRecPoszero[id]['layer']=round*10+actlayer
                                    GameTreeRecPoszero[id]['games'].append(data['hand_id'])
                                    GameTreeRecPoszero[id]['parent']=currentnode
                                    GameTreeRecPoszero[id]['grandpa']=GameTreeRecPoszero[currentnode]['parent']
                                    if currentnode not in GameTreeRecPoszero[id]['parlst']:
                                        GameTreeRecPoszero[id]['parlst'].append(currentnode)

                                    if GameTreeRecPoszero[currentnode]['actname'] in ['2','3','4']:
                                        grandpa=GameTreeRecPoszero[id]['grandpa']
                                        GameTreeRecPoszero[id]['actseq']=GameTreeRecPoszero[grandpa]['actseq']+GameTreeRecPoszero[currentnode]['actname']+actname
                                    else:
                                        GameTreeRecPoszero[id]['actseq']=GameTreeRecPoszero[currentnode]['actseq']+actname

                                    currentnode=id
                                    flag_child_exist=True
                                    if flag_output: print('child found',currentnode,GameTreeRecPoszero[currentnode])
                                    break

                    if not flag_child_exist:#若当前节点不存在子节点时，创建一个新节点
                        newnodeid='node'+str(len(GameTreeRecPoszero))
                        newnodeactname=actname
                        newnodeaccess=1
                        newnodelayer=round*10+actlayer
                        newnodeparent=currentnode
                        newnodegrandpa=GameTreeRecPoszero[currentnode]['parent']
                        newnodeparlst=[currentnode]
                        newnodechild=[]

                        newnodeactseq=''
                        if GameTreeRecPoszero[currentnode]['actname'] in ['2','3','4']:
                            #print('parent',currentnode,GameTreeRecPoszero[currentnode])
                            #print('grandpa',newnodegrandpa,GameTreeRecPoszero[newnodegrandpa])
                            newnodeactseq=GameTreeRecPoszero[newnodegrandpa]['actseq']+GameTreeRecPoszero[currentnode]['actname']+actname
                        else:
                            newnodeactseq=GameTreeRecPoszero[currentnode]['actseq']+actname

                        GameTreeRecPoszero[newnodeid]={'id':newnodeid,'actname':newnodeactname,
                        'access':newnodeaccess,'games':[data['hand_id']],'parent':newnodeparent,
                        'grandpa':newnodegrandpa,'child':newnodechild,'actseq':newnodeactseq,
                        'layer':newnodelayer,
                        'parlst':newnodeparlst,
                        'actpos':actpos}
                        GameTreeRecPoszero[currentnode]['child'].append(newnodeid)
                        if flag_output: print('child build1',newnodeid,GameTreeRecPoszero[newnodeid])
                        currentnode=newnodeid

                if flag_output:
                    print('press any key to continue:')
                    anykey=input()

            #一轮结束后
            round+=1
            if round>4 or GameTreeRecPoszero[currentnode]['actname']=='f' or actdict['mynowbet']>=20000:
                pass
            else:
                actname=str(round)

                flg_round_exist=False
                if GameTreeRecPoszero[currentnode]['child']:
                    actnamechilds=[GameTreeRecPoszero[id]['actname'] for id in GameTreeRecPoszero[currentnode]['child']]
                    if actname in actnamechilds:
                        for id in GameTreeRecPoszero[currentnode]['child']:
                            if GameTreeRecPoszero[id]['actname']==actname:
                                GameTreeRecPoszero[id]['access']+=1
                                GameTreeRecPoszero[id]['layer']=round*10
                                GameTreeRecPoszero[id]['games'].append(data['hand_id'])
                                GameTreeRecPoszero[id]['parent']=currentnode
                                GameTreeRecPoszero[id]['grandpa']=GameTreeRecPoszero[currentnode]['parent']
                                if currentnode not in GameTreeRecPoszero[id]['parlst']:
                                    GameTreeRecPoszero[id]['parlst'].append(currentnode)
                                currentnode=id
                                flg_round_exist=True
                                if flag_output: print('child found-round',currentnode,GameTreeRecPoszero[currentnode])
                                break
                if not flg_round_exist:
                    #一种方式式完全的新增节点
                    #newnodeid='node'+str(len(GameTreeRecPoszero))
                    #另一种方式是判断后用已有的节点
                    if flg_round[round-1]:
                        id=node_round[round-1]
                        GameTreeRecPoszero[currentnode]['child'].append(id)
                        GameTreeRecPoszero[id]['layer']=round*10
                        GameTreeRecPoszero[id]['access']+=1
                        GameTreeRecPoszero[id]['games'].append(data['hand_id'])
                        GameTreeRecPoszero[id]['parent']=currentnode
                        GameTreeRecPoszero[id]['grandpa']=GameTreeRecPoszero[currentnode]['parent']
                        if currentnode not in GameTreeRecPoszero[id]['parlst']:
                            GameTreeRecPoszero[id]['parlst'].append(currentnode)
                        currentnode=id
                        if flag_output: print('child found-round-saved',currentnode,GameTreeRecPoszero[currentnode])
                    else:
                        newnodeid='node'+str(len(GameTreeRecPoszero))
                        newnodeactname=actname
                        newnodeaccess=1
                        newnodelayer=round*10
                        newnodeparent=currentnode
                        newnodegrandpa=GameTreeRecPoszero[currentnode]['parent']
                        newnodeparlst=[currentnode]
                        newnodechild=[]
                        
                        GameTreeRecPoszero[newnodeid]={'id':newnodeid,'actname':newnodeactname,
                        'access':newnodeaccess,'games':[data['hand_id']],
                        'grandpa':newnodegrandpa,'parlst':newnodeparlst,
                        'layer':newnodelayer,
                        'parent':newnodeparent,'child':newnodechild}
                        flg_round[round-1]=True
                        node_round[round-1]=newnodeid
                        print('*********flg_round=',flg_round)
                        print('*********node_round=',node_round)
                        GameTreeRecPoszero[currentnode]['child'].append(newnodeid)
                        if flag_output: print('child build1-round',newnodeid,GameTreeRecPoszero[newnodeid])
                        currentnode=newnodeid

            if flag_output:
                print('press any key to continue:')
                anykey=input()


        #print('update GameTreeRecPoszero=',GameTreeRecPoszero)

    return None




#从读取的每一局数据构建我方处于小盲注位和大盲注位的两张博弈历史树图
def actiontreegen(nstt,nend):
    global GameTreeRecPoszero,GameTreeRecPosones
    global datausers,recAllresults
    global flg_round,node_round

    #初始化
    GameTreeRecPoszero={}
    GameTreeRecPosones={}
    

    #对每局数据做处理
    dataused=[]
    if datausers:
        dataused=datausers
    else:
        dataused=recAllresults

    #构建我方位于小盲注，而对手位于大盲注时的博弈历史树
    flg_round=[False,False,False,False]
    node_round=[None,None,None,None]
    for i in range(nstt,nend):
        data=dataused[i]
        actionhistreezero(data)
        if (i==0 or i==1) and data['position']==0 :
            print(f'{i=:}',data)
            #print('press anykey to continue:')
            #anykey=input()


    #我方位于大盲注，而对手位于小盲注时的博弈历史树
    flg_round=[False,False,False,False]
    node_round=[None,None,None,None]
    for i in range(nstt,nend):
        data=dataused[i]
        actionhistreeones(data)
        if (i==0 or i==1) and data['position']==1 :
            print(f'{i=:}',data)
            #print('press anykey to continue:')
            #anykey=input()
    
    print(f'-----data:{nstt} to {nend}-----')
    print('GameTreeRecPoszero=')
    for k,v in GameTreeRecPoszero.items():
        print('\n',k,":",v)
        break #显示一个
    print('GameTreeRecPosones=')
    for k,v in GameTreeRecPosones.items():
        print('\n',k,":",v)
        break #显示一个

    return None


#将所有的局的链获取出来
#因为链不是唯一的，一轮变化以后就会有新的变化。比如第二轮完全相同，但第一轮是不同的，这样就会存在问题。
def actiontreecluster():
    global GameTreeRecPoszero,GameTreeRecPosones

    dataused=[]
    if datausers:
        dataused=datausers
    else:
        dataused=recAllresults

    TreeChainRec={}
    nodeactall=[]  #全局的所有的情况的记录
    nodeidsall=[]
    nodehidall=[]
    nodenumall=[]
    nodechnall=[]
   
    for id in GameTreeRecPosones: #key 遍历
        if not GameTreeRecPosones[id]['child']: #遍历叶子节点
            print('\nleaf node:',id) #每个叶子节点代表了一个链
            print('act sequence=',GameTreeRecPosones[id]['actseq'])
            print('access number=',GameTreeRecPosones[id]['access'])
            print('access games=',GameTreeRecPosones[id]['games'])
            nodeactseqlst=[]
            nodeidseqlst=[]
            nodehandidlst=[]

            #根据对应的局id，记录所有的可能出现这个叶子节点的情况
            for hand_id in GameTreeRecPosones[id]['games']:#考虑这些局
                currentid=id
                previouid=None
                nodeactsequence=[] #行动序列
                nodeidsequence=[]  #节点id序列
                while True:
                    nodeactsequence.append(GameTreeRecPosones[currentid]['actname'])
                    nodeidsequence.append(GameTreeRecPosones[currentid]['id'])
                    if currentid=='node0':
                        break
                    
                    #往前推，之前的节点变为当前的节点，当前节点变为当前的父节点
                    previouid=currentid
                    currentid=GameTreeRecPosones[currentid]['parent']

                    if GameTreeRecPosones[currentid]['actname'] in ['2','3','4']:
                        #不直接从父节点走，因为父节点记录的是最后一次的父节点，实际父节点可能有多个
                        #GameTreeRecPosones[currentid]['parent']=GameTreeRecPosones[previouid]['grandpa']
                        for parid in GameTreeRecPosones[currentid]['parlst']:
                            #做判断当前局是否在parid节点内
                            if hand_id in GameTreeRecPosones[parid]['games']:
                                GameTreeRecPosones[currentid]['parent']=parid
                                break

                nodeactsequence.reverse()
                nodeidsequence.reverse()
                #print('nodeactsequence=',nodeactsequence)
                #print('nodeidsequence=',nodeidsequence)

                if nodeactsequence not in nodeactseqlst:
                    nodeactseqlst.append(nodeactsequence)
                    nodeidseqlst.append(nodeidsequence)
                    nodehandidlst.append([hand_id])
                else:
                    nodehandidlst[nodeactseqlst.index(nodeactsequence)].append(hand_id)


            #记录当前叶子节点的每一种链的情况
            maxaccess=GameTreeRecPosones['node0']['access']

            print('nodeactseqlst=',nodeactseqlst)
            print('nodeidseqlst=',nodeidseqlst)
            print('nodehandidlst=',nodehandidlst)

            for x in nodeactseqlst:
                nodeactall.append(x) #全局的记录
                nodechnall.append(complementchain(x)) #格式化的补全的链
            for x in nodeidseqlst:
                nodeidsall.append(x)
            for x in nodehandidlst:
                nodehidall.append(x)
                nodenumall.append([len(x),len(x)/maxaccess])


            #把每条链的牌和赢率信息进行输出
            #主要根据handid来获取
            for chainid in range(len(nodeactseqlst)):
                nodehandids=nodehandidlst[chainid]
                nodeactsequence=nodeactseqlst[chainid]

                print('nodehandids=',nodehandids) #当前这种链的所有的局的id
                for hand_id in nodehandids:
                    gameinfocards=[0]*len(nodeactsequence)
                    gameinfowr=[0]*len(nodeactsequence)
                    gameinfobetnow=[0]*len(nodeactsequence)
                    gameinfobetpre=[0]*len(nodeactsequence)
                    gameinfobetopp=[0]*len(nodeactsequence)

                    actionhistory=[] #flat所有的动作到一个列表方便后面取用
                    for x in dataused[hand_id]['action_history']:
                        for y in x:
                            actionhistory.append(y)
                    #print('actionhistory=',actionhistory)

                    
                    i=0 #标记nodeactsequence中的序号
                    j=0 #标记位置
                    k=0 #标记公共牌的位置
                    i1=0 #标记实际第几个动作
                    print('nodeactsequence=',nodeactsequence,len(nodeactsequence)) #输出动作序列
                    for act in nodeactsequence:
                        if act =='1':
                            j=0
                            k=0
                        elif act =='2':
                            j=1
                            k=3
                        elif act =='3':
                            j=1
                            k=4
                        elif act =='4':
                            j=1
                            k=5
                        else:
                            if k>0:
                                gameinfocards[i]=dataused[hand_id]['player_card'][j]+dataused[hand_id]['public_card'][:k]
                            else:
                                gameinfocards[i]=dataused[hand_id]['player_card'][j]
                            j=1-j
                            #print('i=',i,' i1=',i1)
                            gameinfobetnow[i]=actionhistory[i1]['mynowbet']
                            gameinfobetpre[i]=actionhistory[i1]['myprebet']
                            gameinfobetopp[i]=actionhistory[i1]['opprebet']
                            i1+=1
                        i+=1
                    print('info:cards:',hand_id,gameinfocards)
                    
                    for x in range(len(gameinfocards)):
                        if gameinfocards[x]!=0:
                            gameinfowr[x]="{:.3f}".format(getwinrate(2,gameinfocards[x][:2],gameinfocards[x][2:]))
                        
                    print('info:wr   :',hand_id,gameinfowr)
                    print('info:pre-b:',hand_id,gameinfobetpre)
                    print('info:opbet:',hand_id,gameinfobetopp)
                    print('info:mybet:',hand_id,gameinfobetnow)

                    break #只输出一局的信息看，所有直接break了


            
            #print('press anykey to continue')
            #anykey=input()

    '''
    #输出所有链看一下
    i=0
    for i in range(len(nodeactall)):#
        print('\nchain {}:'.format(i))
        print('actseq1:',nodeactall[i])
        print('actseq2:',''.join(nodeactall[i]))
        print('gamesid:',nodehidall[i])
        print('gamesid:',nodechnall[i])
        i+=1

    #将链补全并计算差异度
    chain1=complementchain(nodeactall[1])
    for i in range(6):
        chain2=complementchain(nodeactall[i])
        distance=distanceCOS(chain1,chain2)
        print('distanceCOS=',distance)
        distance=distanceMHD(chain1,chain2)
        print('distanceMHD=',distance)
    '''

    TreeChainRec['TreeRecPosones']=copy.deepcopy(GameTreeRecPosones)
    TreeChainRec['Treechainnodes']=nodeidsall
    TreeChainRec['Treechainacts']=nodeactall
    TreeChainRec['Treechainchns']=nodechnall
    TreeChainRec['Treechainnums']=nodenumall
    TreeChainRec['Treechainhids']=nodehidall

    return TreeChainRec




#比较两个树的相似度
def comparehistree(tree1,tree2):

    m1=len(tree1['Treechainchns'])
    m2=len(tree2['Treechainchns'])
    matrixsimtrees=np.zeros((m1,m2))
    

    for i in range(m1):
        for j in range(m2):
            chain1=tree1['Treechainchns'][i]
            chain2=tree2['Treechainchns'][j]
            distance=distanceCOS(chain1,chain2)
            print('distanceCOS=',distance)
            matrixsimtrees[i,j]=distance
            #distance=distanceMHD(chain1,chain2)
            #print('distanceMHD=',distance)

    print('matrixsimtrees=',matrixsimtrees)

    #先根据最小相似度做排序，并记录排序信息
    if m1<=m2:
        mindis=np.min(matrixsimtrees,axis=1)
        #用列表做排序比较方便于记录
        #在行里头做排序
        mindislst=[[i,mindis[i]] for i in range(len(mindis))]
        mindislstsort=sorted(mindislst,key=lambda list: list[1])
        
    else:
        #在列里头做排序
        mindis=np.min(matrixsimtrees,axis=0)
        mindislst=[[i,mindis[i]] for i in range(len(mindis))]
        mindislstsort=sorted(mindislst,key=lambda list: list[1])

    print('m1=',m1)
    print('m2=',m2)
    print('mindis',mindis,len(mindis))
    print('mindislstsort',mindislstsort)


    if m1<=m2:
        jset=[]
        for x in mindislstsort:
            i=x[0]
            matrixsimi=matrixsimtrees[i,:]
            matrixsimi[jset]=2 #设置一个大的值避免被取到
            j=np.argmin(matrixsimi)
            print('distance=',matrixsimtrees[i,j])
            print('chain i=',tree1['Treechainchns'][i],tree1['Treechainnums'][i])
            print('chain j=',tree2['Treechainchns'][j],tree2['Treechainnums'][j])
            jset.append(j)
    else:
        iset=[]
        for x in mindislstsort:
            j=x[0]
            matrixsimj=matrixsimtrees[:,j]
            matrixsimj[iset]=2 #设置一个大的值避免被取到
            i=np.argmin(matrixsimj)
            print('distance=',matrixsimtrees[i,j])
            print('chain i=',tree1['Treechainchns'][i],tree1['Treechainnums'][i])
            print('chain j=',tree2['Treechainchns'][j],tree2['Treechainnums'][j])
            iset.append(i)

    return None



#计算向量x，y的余弦相似度
def distanceCOS(x1,y1):
    x=np.array(x1)
    y=np.array(y1)
    dotxy=np.dot(x,y)
    lensx=np.sqrt(np.dot(x,x))
    lensy=np.sqrt(np.dot(y,y))
    sim=dotxy/(lensx*lensy)
    if sim>1:
        sim=1
    dis=1-sim
    return dis

#计算向量x，y的曼哈顿距离
def distanceMHD(x1,y1):
    x=np.array(x1)
    y=np.array(y1)
    dxy=np.sum(np.abs(x-y))
    dis=dxy/len(x1)
    return dis


#用1个长度为20的列表来表示动作的历史
#根据输入的链补全这个列表
def complementchain(chainorg):

    print('chain-org:',chainorg)
    chaincpm=[0]*20
    j=0
    for x in chainorg:
        if x in ['1','2','3','4']:
            j=(int(x)-1)*5
            #轮次信息不再给出，因为位置已经给出了轮次信息
            #chaincpm[j]=int(x)
            #j+=1
        else:
            if x=='f':
                chaincpm[j]=1
            elif x=='c':
                chaincpm[j]=2
            elif x=='r':
                chaincpm[j]=3
            elif x=='a':
                chaincpm[j]=4
            j+=1
    print('chain-cpm:',chaincpm)

    return chaincpm



#转成博弈历史树图-同质图(或异质图，这里没有使用节点的特征来区分，所以还是使用同质图)
#与drawactiontree函数类似，只是不绘制图片
#主要用于数据集生成
def featureactiontree(GameTreeRecPos,myposSB=True,tagetlabel=0):
    
    actnamemap={'f':1,'c':2,'r':3,'a':4}
    graphdata = torch_geometric.data.Data() #使用pyg的数据结构，构建图
    nodemap = {key: i for i, key in enumerate(GameTreeRecPos)}
    features=torch.zeros((len(GameTreeRecPos),7)) #节点的属性矩阵

    #在图中添加节点，并按顺序记录节点的颜色，大小，标签等信息
    maxaccess=GameTreeRecPos['node0']['access']
    for k,v in GameTreeRecPos.items():
        feature=torch.zeros(7)  #单个节点的特征
        if GameTreeRecPos[k]['actname'] in ['1','2','3','4']: #轮次起始节点
            '''
            (1) 节点行动类型：0,1,2,3,4：分别表示轮次起始节点，fold，call/check，raise，allin节点\\
            (2) 节点行动者位置：0,1,2：分别表示轮次起点，小盲注位行动，大盲注位行动\\
            (3) 节点行动者类型：0,1,2：分别表示轮次起点，我方行动，对手行动\\
            (4) 轮次：1,2,3,4：分别表示4个轮次\\
            (5) 是轮次第几个行动：用数字表示，轮次起始是0，其它节点则从1开始计数行动\\
            (6) 当前轮次该行动前的加注动作次数：用数字表示，次数从0开始计数\\
            (7) 观测到的次数占比：是实数，为观测到次数与根节点次数的比例
            '''
            feature[0]=0  
            feature[1]=0  
            feature[2]=0  
            feature[3]=int(str(v['layer'])[0])
            feature[4]=0
            feature[5]=0
            feature[6]=v['access']/maxaccess
        elif GameTreeRecPos[k]['actpos']==0: 

            feature[0]=actnamemap[v['actname']]  #
            feature[1]=v['actpos']+1
            
            feature[3]=int(str(v['layer'])[0])
            feature[4]=int(str(v['layer'])[1])
            roundactstr=re.sub("\d","|",v['actseq']).split('|')[-1][:-1]
            feature[5]=roundactstr.count('r')+roundactstr.count('a')
            feature[6]=v['access']/maxaccess

            if myposSB: #当前位置是0，且我是小盲注，则是我方
                feature[2]=1
            else: #当前位置是0，且我是大盲注，则是对手
                feature[2]=2
        else: #pos ==1
            feature[0]=actnamemap[v['actname']]  #
            feature[1]=v['actpos']+1
            
            feature[3]=int(str(v['layer'])[0])
            feature[4]=int(str(v['layer'])[1])
            roundactstr=re.sub("\d","|",v['actseq']).split('|')[-1][:-1]
            feature[5]=roundactstr.count('r')+roundactstr.count('a')
            feature[6]=v['access']/maxaccess
        
            if myposSB: #当前位置是1，且我是小盲注，则是对手
                feature[2]=2
            else: #当前位置是1，且我是大盲注，则是我方
                feature[2]=1
        
        features[nodemap[k],:]=feature[:]
        #print('feature of node=',k,nodemap[k],feature)

    #print('features of graph=',features)

    #在图中添加边，并按顺序记录边的宽度、标签等信息
    edgesrc=[]
    edgedst=[]
    for k,v in GameTreeRecPos.items():
        if v['child']:
            for x in v['child']:
                #print('path=',k,x)
                edgesrc.append(nodemap[k]) #起点索引
                edgedst.append(nodemap[x]) #终点索引

    edge_index = torch.tensor([edgesrc, edgedst])
    #print('edges of graph=',edge_index)

    graphdata.x = features
    graphdata.edge_index=edge_index
    graphdata.y = tagetlabel #图的类别
    #print('graphdata=',graphdata)

    return graphdata




#绘制博弈历史树图
#这是图数据已经有了情况下的绘图
def drawfeatureactiontree(graphdata):
    
    plt.figure(figsize=(10,8))
    G = nx.DiGraph()
    edge_index = graphdata.edge_index.t() #data['edge_index'].t()
    edge_index = np.array(edge_index.cpu())
    features=graphdata.x
    #需要显式的加入节点，否则后面nodecolors顺序不对
    #因为只用add_edges_from的形式加入边，会导致节点的顺序会根据edges
    #的出现而加入从而使得节点不是0到最后一个节点。
    G.add_nodes_from(list(range(len(features)))) 
    G.add_edges_from(edge_index)

    #actnamemap={'f':1,'c':2,'r':3,'a':4}
    actnamemaprev={1:'f',2:'c',3:'r',4:'a'}

    #在图中添加边，并按顺序记录边的宽度、标签等信息
    edgelabels={}
    for edgesrc,edgedst in edge_index.tolist():
        if int(features[edgedst,0].item()) != 0:
            edgelabels[(edgesrc,edgedst)]=actnamemaprev[int(features[edgedst,0].item())]
            print(f'{edgesrc=:},{edgedst=:}')
            print('actname=',edgelabels[(edgesrc,edgedst)])


    #为各个节点添加颜色和大小
    nodelabels={}
    nodecolors=[]
    nodesizes=[]
    i=0
    for feature in features:
        print(f'node {i} feature=',feature)
        if int(feature[2].item())==0:
            ndcolor=2
            nodecolors.append(colors_use[2]) #轮次起始节点
        elif int(feature[2].item())==1:
            ndcolor=0
            nodecolors.append(colors_use[0]) #我方颜色即0
        elif int(feature[2].item())==2:
            ndcolor=1
            nodecolors.append(colors_use[1]) #对手的颜色用1
        nodesizes.append(int(100+feature[6]*200))
        nodelabels[i]=str(i)
        print(f'node {i} label=',nodelabels[i], ' color=',ndcolor)
        i+=1

    #根据第二个节点的是我方还是对手来判断对手是SB还是BB
    
    norignodes=(features[:,-1]>=1).sum().item()
    poslabel=''
    if norignodes>=2:
        poslabel='BB&SB'
    else:
        if features[1,2]==1: #第一个动作是我方，则我方是sb，对手是BB
            poslabel='BB'
        else:
            poslabel='SB'
    plt.title(f'Game historty tree when opponent at {poslabel}')
    
    
    pos=nx.nx_agraph.graphviz_layout(G, prog='dot')  #"dot",patchwork,fdp,sfdp

    nx.draw(G,pos,with_labels=True,labels=nodelabels,
    #width=edgewidths,
    #connectionstyle='Angle3,angleA=90,angleB=0', #'arc3, rad=0.2',angleA=90,angleB=0,
    node_color=nodecolors,node_size=nodesizes, 
    font_size=8,
    #font_weight='bold'
    ) 
    
    nx.draw_networkx_edge_labels(G, pos, #cmap=plt.cm.Blues,
    #label_pos=0.25,
    #horizontalalignment='right',
    #verticalalignment='bottom',
    edge_labels=edgelabels,
    font_size=8,
    #font_weight='bold'
    ) #给边绘制标签
    

    # Make legend,自制的legend
    plt.plot([], [], 'o', markersize=10,color=colors_use[2], label = "Round start")
    plt.plot([], [], 'o', markersize=10,color=colors_use[1], label = "Opponent")
    plt.plot([], [], 'o', markersize=10,color=colors_use[0], label = "Player")
    plt.legend(labelspacing = 1,  frameon = True) #loc='center left', bbox_to_anchor=(0.9, 0.5),
    plt.savefig(f'fig-game-type-nodes-{poslabel}.pdf')
    
    #plt.show()

    return None






#绘制博弈历史树图
#输入数据是从原始的自定义的树结构数据，区分小盲注和大盲注两个位置
def drawactiontree(GameTreeRecPos,myposSB=True,tagetlabel=0,flgchain=False,addmark=""):
    
    treegraph = nx.DiGraph()
    nodelabels={}
    nodecolors=[]
    #nodeshapes=''
    nodesizes=[]
    nodesizes1=[]
    edgelabels={}

    actnamemap={'f':1,'c':2,'r':3,'a':4}

    graphdata = torch_geometric.data.Data() #使用pyg的数据结构，构建图
    nodemap = {key: i for i, key in enumerate(GameTreeRecPos)}
    features=torch.zeros((len(GameTreeRecPos),7)) #节点的属性矩阵

    #在图中添加节点，并按顺序记录节点的颜色，大小，标签等信息
    maxaccess=GameTreeRecPos['node0']['access']
    for k,v in GameTreeRecPos.items():
        feature=torch.zeros(7)  #单个节点的特征
        if GameTreeRecPos[k]['actname'] in ['1','2','3','4']:
            nodecolors.append(colors_use[2])
            feature[0]=0  #轮次起始节点
            feature[1]=0  
            feature[2]=0  
            feature[3]=int(str(v['layer'])[0])
            feature[4]=0
            feature[5]=0
            feature[6]=v['access']/maxaccess
        elif GameTreeRecPos[k]['actpos']==0: 

            feature[0]=actnamemap[v['actname']]  #
            feature[1]=v['actpos']+1
            
            feature[3]=int(str(v['layer'])[0])
            feature[4]=int(str(v['layer'])[1])
            roundactstr=re.sub("\d","|",v['actseq']).split('|')[-1][:-1]
            feature[5]=roundactstr.count('r')+roundactstr.count('a')
            feature[6]=v['access']/maxaccess

            if myposSB: #当前位置是0，且我是小盲注，则使用我方颜色即0
                nodecolors.append(colors_use[0])
                feature[2]=1
            else: #当前位置是0，且我是大盲注，则使用对手颜色即1
                nodecolors.append(colors_use[1]) #对手的颜色用1
                feature[2]=2
        else: #pos ==1
            feature[0]=actnamemap[v['actname']]  #
            feature[1]=v['actpos']+1
            
            feature[3]=int(str(v['layer'])[0])
            feature[4]=int(str(v['layer'])[1])
            roundactstr=re.sub("\d","|",v['actseq']).split('|')[-1][:-1]
            feature[5]=roundactstr.count('r')+roundactstr.count('a')
            feature[6]=v['access']/maxaccess
        
            if myposSB: #当前位置是1，且我是小盲注，则使用对手颜色即1
                nodecolors.append(colors_use[1]) #对手的颜色用1
                feature[2]=2
            else: #当前位置是1，且我是大盲注，则使用我方颜色即0
                nodecolors.append(colors_use[0]) 
                feature[2]=1
        
        features[nodemap[k],:]=feature[:]
        print('feature of node=',k,nodemap[k],feature)

        nodesizes.append(int(100+v['access']/maxaccess*200))
        nodesizes1.append(int(300+v['access']/maxaccess*200))
        treegraph.add_node(k)
        nodelabels[k]=nodemap[k]     #GameTreeRecPos[k]['actname']
    
    print('features of graph=',features)

    
    #在图中添加边，并按顺序记录边的宽度、标签等信息
    edgewidths=[]
    edgesrc=[]
    edgedst=[]
    for k,v in GameTreeRecPos.items():
        if v['child']:
            for x in v['child']:
                #print('path=',k,x)
                treegraph.add_edge(k,x)

                edgesrc.append(nodemap[k]) #起点索引
                edgedst.append(nodemap[x]) #终点索引

                if v['actname']=='a' or GameTreeRecPos[x]['actname']=='a':
                    edgewidths.append(0.5)
                else:
                    edgewidths.append(0.5)
                if GameTreeRecPos[x]['actname'] not in ['2','3','4']:
                    edgelabels[(k,x)]=GameTreeRecPos[x]['actname']

    edge_index = torch.tensor([edgesrc, edgedst])
    print('edges of graph=',edge_index)

    graphdata.x = features
    graphdata.edge_index=edge_index
    graphdata.y = tagetlabel #图的类别,标签
    print('graphdata=',graphdata)


    plt.figure()
    G = nx.Graph()
    edge_index = graphdata.edge_index.t() #data['edge_index'].t()
    edge_index = np.array(edge_index.cpu())
    G.add_edges_from(edge_index)
    #nodepos=nx.kamada_kawai_layout(G) #nx.planar_layout(G) #nx.circular_layout(G)
    nx.draw(G)


    #绘图
    plt.figure()#figsize=(10,8)
    
    
    if flgchain:
        pos=nx.nx_agraph.graphviz_layout(treegraph, prog='patchwork')  #"dot",patchwork,fdp,sfdp
    else:
        pos=nx.nx_agraph.graphviz_layout(treegraph, prog='dot')

    #pos = graphviz_layout(treegraph, prog="dot")
    #pos= nx.nx_pydot.pydot_layout(treegraph, prog='dot')#'twopi
    #pos= nx.spring_layout(treegraph)
    #pos= nx.multipartite_layout(treegraph) #需要subset属性
    #pos=nx.planar_layout(treegraph) 
    #kamada_kawai_layout,bipartite_layout,arf_layout,circular_layout
    #fruchterman_reingold_layout,shell_layout,spring_layout,spectral_layout
    #spiral_layout,planar_layout
    
    #nx.draw(treegraph,with_labels=True,labels=nodelabels,node_color=nodecolors)
    #plt.legend()

    
    nx.draw(treegraph,pos,with_labels=True,labels=nodelabels,
    width=edgewidths,
    #connectionstyle='Angle3,angleA=90,angleB=0', #'arc3, rad=0.2',angleA=90,angleB=0,
    node_color=nodecolors,node_size=nodesizes, 
    font_size=8,
    #font_weight='bold'
    ) 

    
    nx.draw_networkx_edge_labels(treegraph, pos, #cmap=plt.cm.Blues,
    #label_pos=0.25,
    #horizontalalignment='right',
    #verticalalignment='bottom',
    edge_labels=edgelabels,
    font_size=8,
    #font_weight='bold'
    ) #给边绘制标签
    


    # Make legend,自制的legend
    flg_figcn=True
    if flg_figcn:
        if myposSB:
            if flgchain:
                plt.title('对手在BB时的博弈历史链',fontsize=18)
            else:
                plt.title('对手在BB时的博弈历史树',fontsize=18)
        else:
            if flgchain:
                plt.title('对手在SB时的博弈历史链',fontsize=18)
            else:
                plt.title('对手在SB时的博弈历史树',fontsize=18)
        plt.plot([], [], 'o', markersize=10,color=colors_use[2], label = "轮次起点(Round start)")
        plt.plot([], [], 'o', markersize=10,color=colors_use[1], label = "对手(Opponent)")
        plt.plot([], [], 'o', markersize=10,color=colors_use[0], label = "参与者(Player) P")
        if flgchain:
            plt.legend(frameon = False, fontsize=14) #loc='center left', bbox_to_anchor=(0.8, 0.6),, loc='center right', bbox_to_anchor=(1.1, 0.75)
        else:
            plt.legend(frameon = False, fontsize=14) #labelspacing = 1,
        plt.tight_layout()
        plt.savefig('fig-game-type-nodes-{}-{}-{}-cn.pdf'.format('SB' if myposSB else 'BB','chain' if flgchain else 'tree',addmark))
        plt.savefig('fig-game-type-nodes-{}-{}-{}-cn.svg'.format('SB' if myposSB else 'BB','chain' if flgchain else 'tree',addmark),format='svg')

    else:
        #注意标题plt.title必须在nx.draw之前给出绘制才行，否则无法显示，比如方法这里的就无法显示。
        if myposSB:
            if flgchain:
                plt.title('Game history chain when opponent at BB',fontsize=18)
            else:
                plt.title('Game history tree when opponent at BB',fontsize=18)
        else:
            if flgchain:
                plt.title('Game history chain when opponent at SB',fontsize=18)
            else:
                plt.title('Game history tree when opponent at SB',fontsize=18)
        plt.plot([], [], 'o', markersize=10,color=colors_use[2], label = "Round start")
        plt.plot([], [], 'o', markersize=10,color=colors_use[1], label = "Opponent")
        plt.plot([], [], 'o', markersize=10,color=colors_use[0], label = "Player P")
        if flgchain:
            plt.legend(frameon = False, fontsize=14, loc='center right', bbox_to_anchor=(1.1, 0.75)) #loc='center left', bbox_to_anchor=(0.8, 0.6),
        else:
            plt.legend(frameon = False, fontsize=14) #labelspacing = 1,
        plt.tight_layout()
        plt.savefig('fig-game-type-nodes-{}-{}-{}.pdf'.format('SB' if myposSB else 'BB','chain' if flgchain else 'tree',addmark))
        plt.savefig('fig-game-type-nodes-{}-{}-{}.svg'.format('SB' if myposSB else 'BB','chain' if flgchain else 'tree',addmark),format='svg')


    #无信息的图
    plt.figure()
    nx.draw(treegraph,pos,width=edgewidths,node_color=nodecolors,node_size=nodesizes1, 
    ) 
    #nx.draw_networkx_edge_labels(treegraph, pos) 
    plt.savefig('fig-game-type-nodes-{}-{}-{}-nonote.svg'.format('SB' if myposSB else 'BB','chain' if flgchain else 'tree',addmark))


    
    '''pydot绘图
    graph = nx.drawing.nx_pydot.to_pydot(treegraph)
    #output_graphviz_svg = graph.create_svg()
    graph.write_png("output.png")
    '''

    '''
    for k,v in GameTreeRecPos.items():
        print(k,' : ',v)
    '''

    graph = pydot.Dot("my_graph", graph_type="digraph", layout='dot')#neato
    #一层层的找：注意到在preflop出现了r超过3次的情况，crrrac，这个可能与acpc服务器的设置相关。
    for layer in [10,11,12,13,14,15,16,20,21,22,23,24,25,30,31,32,33,34,35,40,41,42,43,44,45]:
        for k,v in GameTreeRecPos.items():
            if v['layer']==layer:
                if GameTreeRecPos[k]['actname'] in ['1','2','3','4']:
                    graph.add_node(pydot.Node(k,label=v['actname'],style="filled",
                    fontname="times-bold",
                    width=(0.4+v['access']/maxaccess*0.6),
                    fontcolor='black',fillcolor=colors_use[2],shape="circle",color=colors_use[2]))
                elif GameTreeRecPos[k]['actpos']==0:
                    graph.add_node(pydot.Node(k,label=v['actname'],style="filled",
                    fontname="times-bold",
                    shape="circle",#"box",
                    width=(0.4+v['access']/maxaccess*0.6),
                    fontcolor='black',color=colors_use[1],fillcolor=colors_use[1]))
                else:
                    graph.add_node(pydot.Node(k,label=v['actname'],style="filled",
                    fontname="times-bold",
                    #fontsize=20, #所有的属性可以参考https://graphviz.org/docs/attrs
                    #width=5,#形状的宽度
                    shape="circle",#"triangle",
                    width=(0.4+v['access']/maxaccess*0.6),
                    fontcolor='black',color=colors_use[0],fillcolor=colors_use[0]))

    for k,v in GameTreeRecPos.items():
        if v['child']:
            for x in v['child']:
                graph.add_edge(pydot.Edge(k,x, color=colors_use[4]))

    #graph.set_prog('twopi')
    #graph.write_png("output.png")
    graph.write_pdf('fig-game-type-pydot-nodes-{}.pdf'.format('SB' if myposSB else 'BB'))
    
    
    plt.show()

    return graphdata



#一个的图神经网络分类器
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_graph_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_graph_classes)
        )
        
        

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        #x = global_mean_pool(x, batch)
        #x = global_add_pool(x, batch)
        x = global_max_pool(x, batch) #效果比上述两个好

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        
        return x


class GinNN(torch.nn.Module):

    def __init__(self, num_node_features, hidden_channels, num_graph_classes):
        super(GinNN, self).__init__()
        torch.manual_seed(12345)
        self.gin = torch_geometric.nn.GIN(num_node_features, hidden_channels,4,out_channels=hidden_channels)
        #print('gin=',self.gin)
        self.sagpool=SAGPooling(hidden_channels,0.8)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels*2, num_graph_classes)
        )
        
        

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        #x = global_max_pool(x, batch) #效果比上述两个好
        x = global_add_pool(x, batch)

        #注意力机制的pool
        #x=self.sagpool(x,edge_index,batch=batch)[0]
        #print('x=',x)
        #anykey=input()

        #x = F.dropout(x, p=0.5, training=self.training)
        x = x.relu()
        x = self.classifier(x)
        
        return x


class GINnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(5):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None,dropout=0.5) #, dropout=0.0

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        #x = global_add_pool(x, batch)
        x = global_max_pool(x, batch) #效果比上述两个好
        return self.mlp(x)



class GRFN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.usednodes=20
        self.in_channels=in_channels
        self.mlpA = MLP([in_channels*self.usednodes, hidden_channels, hidden_channels], norm=None,dropout=0.5) #, dropout=0.0

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None,dropout=0.5) #, dropout=0.0

    
    def forward(self, x, edge_index, batch):
        bsize=torch.max(batch).item()+1
        bdata=[]
        for bi in range(bsize):
            xid=torch.nonzero(batch==bi).squeeze()
            xsingle=x[xid]
            _, indices = torch.sort(xsingle,dim=-2,descending=True)
            idx = indices[:, -1] #根据最后一列数据来考虑排序
            ls = []
            for ix in idx[1:self.usednodes+1]: #在从0开始行排序索引中，取排序中第1到第20行，
                ls.append(xsingle[ix])
            nrealnods=len(ls)
            if nrealnods<self.usednodes:
                for _ in range(self.usednodes-nrealnods):
                    ls.append(torch.tensor([0]*self.in_channels))
            xg=torch.cat(ls).unsqueeze(0)
            bdata.append(xg)
        xb=torch.cat(bdata)
        xh=self.mlpA(xb)
        out=self.mlp(xh)
        return out
    

    def forwardtest(self, x, edge_index, batch):
        
        print(x.size())
        print(batch,batch.size())

        bsize=torch.max(batch).item()+1
        print('bsize=',bsize)
        bdata=[]
        for bi in range(bsize):
            xid=torch.nonzero(batch==bi).squeeze()
            print('xid=',xid)
            xsingle=x[xid]
            print('xsingle=',xsingle,xsingle.size())
            _, indices = torch.sort(xsingle,dim=-2,descending=True)
            print('indices=',indices)
            idx = indices[:, -1] #根据最后一列数据来考虑排序
            print('idx=',idx)
            ls = []
            for ix in idx[1:21]: #在从0开始行排序索引中，取排序中第1到第20行，
                ls.append(xsingle[ix])
            nrealnods=len(ls)
            if nrealnods<self.usednodes:
                for _ in range(self.usednodes-nrealnods):
                    ls.append(torch.tensor([0]*self.in_channels))
            xg=torch.cat(ls).unsqueeze(0)
            print('xg=',xg,xg.size())
            bdata.append(xg)
            print('anykeyto continue')
            anykey=input()
        xb=torch.cat(bdata)
        xh=self.mlpA(xb)
        out=self.mlp(xh)
        return out
    
def testGINnetA():
    
    dataset=GameHisTreeDataset("GHTdata3000")
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    model =GRFN(7, 20, 14)
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        print('out=',out)
        break
    
    return None


class GINnetB(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(1):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None,dropout=0.5) #, dropout=0.0

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        #x = global_add_pool(x, batch)
        x = global_max_pool(x, batch) #效果比上述两个好
        return self.mlp(x)

class GINnetC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None,dropout=0.5) #, dropout=0.0

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        #x = global_add_pool(x, batch)
        x = global_max_pool(x, batch) #效果比上述两个好
        return self.mlp(x)


#模型的训练
def train(loader,model,optimizer,criterion):
    model.train()

    lossval=0.0
    i=0
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
       
        loss = criterion(out, data.y)
        lossval+=loss.item()
        i+=1

        loss.backward()
        optimizer.step()
    lossval=lossval/i

    return lossval

#模型的测试
def test(loader,model):
    model.eval()
    
    correct = 0
    errorclass=[]
    for data in loader:                                  # 批遍历测试集数据集。
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch) # 一次前向传播
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == data.y).sum())           # 检查真实标签
        acc=correct / len(loader.dataset)

        errorclass+=data.y[torch.where(torch.abs(data.y-pred))].tolist() #把错误分类的真实类别记录下来
    
    nerrcls=[errorclass.count(i) for i in range(14)]
    return acc,nerrcls


#提取数据并训练图分类器
def testTrainGraphClassifier(ngsamples=3000,nettype='',batchsize=100,epoch_size=100,learnrate=0.001):
    global device

    dataset=GameHisTreeDataset(f"GHTdata{ngsamples}")
    nnmodelfile=f'recog-op-GNN-Ngtr-{ngsamples}{nettype}.pkl'
    print(f"\nDataset: {dataset}:")
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    torch.manual_seed(12345) #种子是固定的
    #所以shuffle出来的序列也是固定的，这样每次训练调用这个函数不会导致训练数据不一样
    dataset = dataset.shuffle() 

    if nettype=='G':
        train_dataset = dataset[len(dataset) // 10*8:]
    elif nettype=='H':
        train_dataset = dataset[len(dataset) // 10*6:]
    elif nettype=='I':
        train_dataset = dataset[len(dataset) // 10*4:]
    else:
        train_dataset = dataset[len(dataset) // 10*2:]
    test_dataset = dataset[:len(dataset) // 10*2]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    
    '''
    torch_geometric提供的批处理方式是：邻接矩阵以对角化方式堆叠（创建一个包含多个独立子图的巨大图），
    节点和目标特征中节点维度简单的连接起来。因此
        依赖于消息传递方案的GNN算子不需要修改，因为消息不在属于不同图的两个节点之间交换。
        由于邻接矩阵以仅包含非零条目（即边）的稀疏方式保存，因此没有计算或内存开销。
    其输出提供了一个batch向量
        它将每个节点映射到批处理中的相应图：batch的长度为一个批的图的节点总数，即为每个节点设定一个图的索引
    '''


    num_node_features=len(test_dataset[0].x[0])
    hidden_channels=64
    num_graph_classes=14
    if nettype=='A':
        model =GRFN(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='B':
        model =GINnetB(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='C':
        model =GINnetC(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='D':
        model =GINnet(num_node_features, 16, num_graph_classes)
    elif nettype=='E':
        model =GINnet(num_node_features, 32, num_graph_classes)
    elif nettype=='F':
        model =GINnet(num_node_features, 80, num_graph_classes)
    elif nettype in ['G','H','I']:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)
    else:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)
    print('model=',model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile,map_location=torch.device('cpu')))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    model=model.to(device)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)


    
    train_acc_his=[]
    test_acc_his=[]
    train_los_his=[]
    best_modelparas={}
    best_trainacc=0.0
    best_testacc=0.0
    tqloop = tqdm.tqdm(range(epoch_size), desc='Epoch') #ncols=100
    for epoch in tqloop:
        #训练
        train_los=train(train_loader,model,optimizer,criterion)
        train_los_his.append(train_los)
        #测试
        train_acc,_ = test(train_loader,model)
        train_acc_his.append(train_acc)

        test_acc,_ = test(test_loader,model)
        test_acc_his.append(test_acc)

        if train_acc> best_trainacc:
            best_modelparas=model.state_dict()
            best_trainacc=train_acc
            best_testacc=test_acc
        elif train_acc== best_trainacc and test_acc> best_testacc:
            best_modelparas=model.state_dict()
            best_testacc=test_acc
        
        tqloop.set_postfix(Acctb=best_testacc,Accb=best_trainacc,los=train_los,Acc=train_acc,Acct=test_acc,refresh=False)
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    
    #保存训练后的模块的参数
    #参数不用转成cpu()也能在后续读取
    torch.save(best_modelparas,nnmodelfile)


    plt.figure()
    epoches=np.linspace(1,epoch_size,epoch_size)
    plt.plot(epoches,train_los_his,label='Los_train')
    plt.plot(epoches,train_acc_his,label='Acc_train')
    plt.plot(epoches,test_acc_his,label='Acc_test')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()

    #测试集的具体情况输出
    batchsize=20
    model=model.to(device)
    model.load_state_dict(best_modelparas)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    test_acc, nerrclass = test(test_loader,model)
    yvalist=[]
    for data in test_loader:
        yvalist+=data.y.tolist() 
    nallclass=[yvalist.count(i) for i in range(num_graph_classes)]
    AccClass=((torch.tensor(nallclass)-torch.tensor(nerrclass))/torch.tensor(nallclass)).tolist()
    print(f'{nallclass=:}')
    #print(f'{nerrclass}')
    print(f'{AccClass=:}')
    print(f'{test_acc=:}')


    return None




#提取数据并测试图分类器
def testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype=''):
    global device
    
    dataset=GameHisTreeDataset(f"GHTdata{ngtestsamples}")
    nnmodelfile=f'recog-op-GNN-Ngtr-{ngtrainsamples}{nettype}.pkl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(12345) #种子是固定的
    #所以shuffle出来的序列也是固定的，这样每次训练调用这个函数不会导致训练数据不一样
    dataset = dataset.shuffle() 

    test_dataset = dataset[:len(dataset) // 10*2]

    num_node_features=len(test_dataset[0].x[0])
    hidden_channels=64
    num_graph_classes=14
    if nettype=='A':
        model =GRFN(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='B':
        model =GINnetB(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='C':
        model =GINnetC(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='D':
        model =GINnet(num_node_features, 16, num_graph_classes)
    elif nettype=='E':
        model =GINnet(num_node_features, 32, num_graph_classes)
    elif nettype=='F':
        model =GINnet(num_node_features, 80, num_graph_classes)
    elif nettype in ['G','H','I']:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)
    else:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)


    '''
    print(f"\nDataset: {dataset}:")
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print('model=',model)
    '''    

    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile,map_location=torch.device('cpu')))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    batchsize=20
    model=model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    test_acc, nerrclass = test(test_loader,model)

    yvalist=[]
    for data in test_loader:
        yvalist+=data.y.tolist() 
    nallclass=[yvalist.count(i) for i in range(num_graph_classes)]

    AccClass=((torch.tensor(nallclass)-torch.tensor(nerrclass))/torch.tensor(nallclass)).tolist()
    
    print(f'{nallclass=:}')
    #print(f'{nerrclass}')
    print(f'{AccClass=:}')
    print(f'{test_acc=:}')
   
    return 1-test_acc,test_acc,AccClass




#训练结果测试与分析
def resultpostdeal(ftout=None):

    ntrainslst=[500,750,1000,3000]
    ntestsplst=[500,750,1000,3000]

    #测试第一个问题
    #训练特征和测试特征来自相同的局数，考察不同特征的对于影响
    Allacccls=[]
    #ntrainslst.reverse()
    for ntrainsamples in ntrainslst:
        ntestsamples=ntrainsamples

        print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
        print('{:^10} {:^10} {:^10} {:^10}'.format('Ngtr','Ngtt','Error','ACC'))
        
        Error,ACC,acccls=testGraphClassifier(ngtrainsamples=ntrainsamples,ngtestsamples=ntestsamples)
        print('{:^10} {:^10} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,Error,ACC),acccls)
        Allacccls.append(acccls)
    
    Allaccclsary=torch.tensor(Allacccls).T.tolist()
    for i in range(len(Allaccclsary)):
        Allaccclsary[i]=[i]+Allaccclsary[i]
    Allaccclsary=[[r'Class\Ng']+ntrainslst]+Allaccclsary
    np.savetxt('ACC-CLASS-SAME-Ng.csv', Allaccclsary, fmt='%s', delimiter=',')


    #测试第二个问题
    #训练特征和测试特征来自不同的局数，考察不同特征对于识别率的影响
    AllaccNeqsps=[ntestsplst]
    for ntrainsamples in ntrainslst:
        AllaccNeqspsrow=[]
        for ntestsamples in ntestsplst:
            
                print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
                print('{:^10} {:^10} {:^10} {:^10}'.format('Ngtr','Ngtt','Error','ACC'))
                
                Error,ACC,acccls=testGraphClassifier(ngtrainsamples=ntrainsamples,ngtestsamples=ntestsamples)
                print('{:^10} {:^10} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,Error,ACC),acccls)
                AllaccNeqspsrow.append(ACC)
        AllaccNeqsps.append(AllaccNeqspsrow)
    
    AllaccNeqsps=torch.tensor(AllaccNeqsps).T.tolist()
    AllaccNeqsps=[[r'Ngtt\Ngtr']+ntrainslst]+AllaccNeqsps
    np.savetxt('ACC-CLASS-Ngtr-Ngtt.csv', AllaccNeqsps, fmt='%s', delimiter=',')

    # 测试第三个问题
    # 更少的训练数据得到的结果
    ntrainslst=[50,100,150,200,250,300, 500,750,1000,3000]
    ntestsplst=[50,100,150,200,250,300, 500,750,1000,3000]
    AllaccNeqsps=[ntestsplst]
    for ntrainsamples in ntrainslst:
        AllaccNeqspsrow=[]
        for ntestsamples in ntestsplst:
            print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
            print('{:^10} {:^10} {:^10} {:^10}'.format('Ngtr','Ngtt','Error','ACC'))
            
            Error,ACC,acccls=testGraphClassifier(ngtrainsamples=ntrainsamples,ngtestsamples=ntestsamples)
            print('{:^10} {:^10} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,Error,ACC),acccls)
            AllaccNeqspsrow.append(ACC)
        AllaccNeqsps.append(AllaccNeqspsrow)
    AllaccNeqsps=torch.tensor(AllaccNeqsps).T.tolist()
    AllaccNeqsps=[[r'Ngtt\Ngtr']+ntrainslst]+AllaccNeqsps
    np.savetxt('ACC-ALL-Ngtr-Ngtt.csv', AllaccNeqsps, fmt='%s', delimiter=',')

    
    # 第四个问题
    if ftout != None and ftout in list(range(7)):
        ftoutACC=[ntrainslst]
        ftoutACCrow=[]
        for ntrainsamples in ntrainslst:
            ntestsamples=ntrainsamples

            print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
            print('{:^10} {:^10} {:^10} {:^10}'.format('Ngtr','Ngtt','Error','ACC'))
            
            Error,ACC,acccls=testGraphClassifier(ngtrainsamples=ntrainsamples,ngtestsamples=ntestsamples)
            print('{:^10} {:^10} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,Error,ACC),acccls)
            ftoutACCrow.append(ACC)
        ftoutACC.append(ftoutACCrow)
        ftoutACC=np.array(ftoutACC).T.tolist()
        ftoutACC=[[r"Ng\No FT", ftout]]+ftoutACC
        np.savetxt(f'ACC-featurout-{ftout}.csv', ftoutACC, fmt='%s', delimiter=',')
    
    return None






#在线时的分类概率测试
#nginter为间隔几局更新一个图
def testOnlineRecog(ngstt=6,nginter=6,ngtrainsamples=3000,nettype=''):
    
    nnmodelfile=f'recog-op-GNN-Ngtr-{ngtrainsamples}{nettype}.pkl'

    num_node_features=7
    hidden_channels=64
    num_graph_classes=14
    if nettype=='A':
        model =GRFN(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='B':
        model =GINnetB(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='C':
        model =GINnetC(num_node_features, hidden_channels, num_graph_classes)
    elif nettype=='D':
        model =GINnet(num_node_features, 16, num_graph_classes)
    elif nettype=='E':
        model =GINnet(num_node_features, 32, num_graph_classes)
    elif nettype=='F':
        model =GINnet(num_node_features, 80, num_graph_classes)
    elif nettype in ['G','H','I']:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)
    else:
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)

    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile,map_location=torch.device('cpu')))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    playername='ASHE'
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    
    model.eval()

    dataset=torch.load(f"GHT-online-{playername}-{opnames[0]}.pt")
    ndata=len(dataset)
    
    j=0
    recoghis=torch.zeros((14,ndata,2))
    recogprobhis=torch.zeros((14,ndata,14))
    recoggamenum=torch.zeros(14)
    for opname in opnames:
        #opname=opnames[0]
        dataset=torch.load(f"GHT-online-{playername}-{opname}.pt")
        print('len(dataset)',len(dataset))

        batchsize=1
        test_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        
        i=0
        for data in test_loader:                             # 批遍历测试集数据集。
            
            out = model(data.x, data.edge_index, data.batch) # 一次前向传播
            #print('out=',out)
            prob= F.softmax(out,dim=1)
            recogprobhis[j,i,:]=prob[0][:]
            #print('prob=',prob)
            yidx=data.y.item()
            #print('yidx=',yidx)
            classprob=prob[0][yidx].item()
            #print('prob[class]={:.6f}'.format(classprob))
            recoghis[j,i,0]=ngstt+i*nginter
            recoghis[j,i,1]=classprob
            i+=1
        j+=1
    
    print('recoghis=',recoghis,recoghis.size())

    for j in range(len(opnames)):
        probgtthreshold=0
        for i in range(1,ndata):
            if (recoghis[j,i,1]>0.9 and recoghis[j,i-1,1]>0.9):
                probgtthreshold+=1
            else:
                probgtthreshold=0
            if probgtthreshold>1:
                recoggamenum[j]=recoghis[j,i,0]
                break
    print('recoggamenum=',recoggamenum)
    np.savetxt(f"res-numbergames-recognized-{ngtrainsamples}.csv",recoggamenum.unsqueeze(1).numpy(), delimiter =",",
        fmt ='%s')



    j=0
    for opname in opnames:
        plt.figure()
        plt.plot(recoghis.numpy()[j,:,0],recoghis.numpy()[j,:,1],
                 label=opname,marker=markline[j],markevery=13,color=colors_use[j])
        '''
        for k in range(14):
            plt.plot(recoghis.numpy()[j,:,0],recogprobhis.detach().numpy()[j,:,k],
                     label="P of "+opnames[k],marker=markline[k],markevery=3,color=colors_use[k])
        '''
        plt.title(f'Recognition of {opname} by RCG{ngtrainsamples}')
        plt.xlabel('Number of games')
        plt.ylabel('Porbability of current opponent')
        plt.ylim(-0.05,1.05)
        plt.xlim(0,3000)
        plt.legend(frameon=False)
        plt.savefig(f'fig-op-recog-{opname}-ngtr-{ngtrainsamples}.pdf')
        
        j+=1


    plt.figure()
    j=0
    for opname in opnames:
        plt.plot(recoghis.numpy()[j,:,0],recoghis.numpy()[j,:,1],
                 label=opname,marker=markline[j],markevery=3,color=colors_use[j])
        j+=1

    plt.title(f'Probability of each opponent')
    plt.xlabel('number of games')
    plt.ylabel('Porbability')
    plt.ylim(-0.05,1.05)
    plt.xlim(0,min([ngtrainsamples*2,500]))
    plt.legend(frameon=False,loc='center right',fontsize=8)
    #plt.show()


    opnameidx={'ElDonatoro':0,'Feste':1,'HITSZ':2,'Hugh_iro':3,'Hugh_tbr':4,'Intermission':5,'PokerBot5':6,
                'PokerCNN':7,'PPPIMC':8,'Rembrant6':9,'RobotShark_iro':10,'RobotShark_tbr':11,'SimpleRule':12,'Slumbot':13}

    '''
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                    'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    '''
    '''
    opnameset1=['ElDonatoro','HITSZ','PokerBot5','PPPIMC','Rembrant6','SimpleRule']
    plt.figure()
    for opname in opnameset1:
        plt.plot(recoghis.numpy()[opnameidx[opname],:,0],recoghis.numpy()[opnameidx[opname],:,1],
                 label=opname,marker=markline[opnameidx[opname]],markevery=3,color=colors_use[opnameidx[opname]])

    plt.title(f'Probability of each opponent with RCG{ngtrainsamples}')
    plt.xlabel('Number of games')
    plt.ylabel('Porbability')
    plt.ylim(-0.05,1.05)
    plt.xlim(0,min([ngtrainsamples*2,3000]))
    plt.legend(frameon=False,loc='center right',fontsize=8)
    plt.savefig(f'fig-op-recog-ngtr-{ngtrainsamples}-set1.pdf')

    opnameset1=['Feste','Hugh_iro','Hugh_tbr','Intermission','PokerCNN']
    plt.figure()
    for opname in opnameset1:
        plt.plot(recoghis.numpy()[opnameidx[opname],:,0],recoghis.numpy()[opnameidx[opname],:,1],
                 label=opname,marker=markline[opnameidx[opname]],markevery=3,color=colors_use[opnameidx[opname]])

    plt.title(f'Probability of each opponent with RCG{ngtrainsamples}')
    plt.xlabel('Number of games')
    plt.ylabel('Porbability')
    plt.ylim(-0.05,1.05)
    plt.xlim(0,min([ngtrainsamples*2,3000]))
    plt.legend(frameon=False,fontsize=8) #,loc='center right'
    plt.savefig(f'fig-op-recog-ngtr-{ngtrainsamples}-set2.pdf')

    opnameset1=['RobotShark_iro','RobotShark_tbr','Slumbot']
    plt.figure()
    for opname in opnameset1:
        plt.plot(recoghis.numpy()[opnameidx[opname],:,0],recoghis.numpy()[opnameidx[opname],:,1],
                 label=opname,marker=markline[opnameidx[opname]],markevery=3,color=colors_use[opnameidx[opname]])

    plt.title(f'Probability of each opponent with RCG{ngtrainsamples}')
    plt.xlabel('Number of games')
    plt.ylabel('Porbability')
    plt.ylim(-0.05,1.05)
    plt.xlim(0,min([ngtrainsamples*2,3000]))
    plt.legend(frameon=False,fontsize=8) #,loc='center right'
    plt.savefig(f'fig-op-recog-ngtr-{ngtrainsamples}-set3.pdf')
    '''

    return None





#识别出当前对手的概率(几个识别器一起画)
#在线时的分类概率测试
#nginter为间隔几局更新一个图
def testOnlineRecogAllclassifier(ngstt=6,nginter=6,ngtrainsampleslst=[50,100,200,250],addfiglabel='A'):
    

    num_node_features=7
    hidden_channels=64
    num_graph_classes=14
    model =GINnet(num_node_features, hidden_channels, num_graph_classes)
    
    #ngtrainsampleslst=[50,100,200,250]
    #addfiglabel='A'

    playername='ASHE'
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                ]
    opnameidx={'ElDonatoro':0,'Feste':1,'HITSZ':2,'Hugh_iro':3,'Hugh_tbr':4,'Intermission':5,'PokerBot5':6,
                'PokerCNN':7,'PPPIMC':8,'Rembrant6':9,'RobotShark_iro':10,'RobotShark_tbr':11,'SimpleRule':12,'Slumbot':13}
    
    dataset=torch.load(f"GHT-online-{playername}-{opnames[0]}.pt")
    ndata=len(dataset)
    recgnum=[ngstt+i*nginter for i in range(ndata)]
    recoghis=torch.zeros((14,ndata,len(ngtrainsampleslst)))

    #遍历所有的识别器
    k=0
    for ngtrainsamples in ngtrainsampleslst:
        nnmodelfile=f'recog-op-GNN-Ngtr-{ngtrainsamples}.pkl'
        try:
            if os.path.exists(nnmodelfile):
                model.load_state_dict(torch.load(nnmodelfile,map_location=torch.device('cpu')))
        except FileNotFoundError:
            print('warning: parameter of the model was not loaded!')
        model.eval()
        
        j=0 #遍历所有对手
        for opname in opnames:
            dataset=torch.load(f"GHT-online-{playername}-{opname}.pt")
            batchsize=1
            test_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
            
            i=0 #遍历所有数据
            for data in test_loader:                             # 批遍历测试集数据集。
                out = model(data.x, data.edge_index, data.batch) # 一次前向传播
                prob= F.softmax(out,dim=1)
                yidx=data.y.item()
                classprob=prob[0][yidx].item()
                recoghis[j,i,k]=classprob
                i+=1
            j+=1
        k+=1
    print('recoghis=',recoghis,recoghis.size())

    '''
    j=0
    for opname in opnames:
        plt.figure()
        for k in range(len(ngtrainsampleslst)):
            plt.plot(recgnum,recoghis.detach().numpy()[j,:,k],
                     label="Classifier-Ng"+str(ngtrainsampleslst[k]),marker=markline[k],
                     linestyle=linestyles[k][1],
                     markevery=3,color=colors_use[j])
        plt.ylim(-0.05,1.05)
        plt.xlim(0,max(ngtrainsampleslst)+50)
        plt.title(f'Recognition by different Classifiers in match against {opname}')
        plt.xlabel('Number of games')
        plt.ylabel('Porbability of correct class')
        plt.legend(frameon=False,loc='lower right',fontsize=8)
        plt.savefig(f'fig-op-recog-{opname}-ngtr-all.pdf')
        
        j+=1
    '''

    opnameset1=['Feste','Hugh_tbr','RobotShark_iro','RobotShark_tbr']
    for opname in opnameset1:
        plt.figure()
        for k in range(len(ngtrainsampleslst)):
            plt.plot(recgnum,recoghis.detach().numpy()[opnameidx[opname],:,k],
                     label="RCG"+str(ngtrainsampleslst[k]),marker=markline[k],
                     linestyle=linestyles[k][1],linewidth=2,
                     markevery=17,color=colors_use[k])
        plt.ylim(-0.05,1.05)
        plt.xlim(0,3000)
        flg_figcn=True
        if flg_figcn:
            plt.title(f'不同识别器对 {opname} 的识别')
            plt.xlabel('局数')
            plt.ylabel('概率')
            #plt.grid()
            plt.legend(frameon=False)
            plt.savefig(f'fig-op-recog-{opname}-ngtr-all-CRG-{addfiglabel}-cn.pdf')
        else:
            plt.title(f'Recognition of {opname} by RCGs')
            plt.xlabel('Number of games')
            plt.ylabel('Porbability of the current opponenet')
            #plt.grid()
            plt.legend(frameon=False)
            plt.savefig(f'fig-op-recog-{opname}-ngtr-all-CRG-{addfiglabel}.pdf')



    #plt.show()


    return None




#单个文件的在线图数据生成
#模拟在线博弈时获得的数据，从第ngstt局开始，每隔nginter局更新一个图，直到ngend局
#用于观察识别器对于类型的判断的概率输出。
#每一类对手的都生成一个数据集，并保存到文件中
#从所有玩家的log文件准备数据
#图数据+对手标签
def prepareSGGDonline(filename,myname,opnamesim,opnamereal,bigblindfirst=False,ngstt=6,ngend=400,nginter=6,ftout=False):
    
    opnames=['ElDonatoro_2pn_2017','Feste_2pn_2017','HITSZ_2pn_2017',
             'Hugh_iro_2pn_2017','Hugh_tbr_2pn_2017','Intermission_2pn_2017',
             'PokerBot5_2pn_2017','PokerCNN_2pn_2017','PPPIMC_2pn_2017','Rembrant6_2pn_2017',
             'RobotShark_iro_2pn_2017','RobotShark_tbr_2pn_2017','SimpleRule_2pn_2017','Slumbot_2pn_2017'
                ]
    
    #当前文件的图数据准备
    datasetgh=[]
    ndata=logfiledealing(filename,myname,bigblindfirst=bigblindfirst)
    for k in range(ngstt,ngend,nginter): #nginter局一个图
        actiontreegen(0,k)
        gsb=featureactiontree(GameTreeRecPoszero,myposSB=True,tagetlabel=opnames.index(opnamereal))  #我方小盲注位时的博弈图
        gbb=featureactiontree(GameTreeRecPosones,myposSB=False,tagetlabel=opnames.index(opnamereal)) #我方大盲注位时的博弈图
        
        #将两个位置的图合并成一个图
        gsbandbb = torch_geometric.data.Data()
        gsbandbb.x=torch.cat([gsb.x,gbb.x],0)

        if  (type(ftout) != type(False)) and (ftout in list(range(7))):
            ftleft=list(range(7))
            ftleft.remove(ftout)
            xftleft=gsbandbb.x[:,ftleft]
            gsbandbb.x=xftleft

        nsbnode=len(gsb.x)
        edgeidxbb=gbb.edge_index+nsbnode
        gsbandbb.edge_index=torch.cat([gsb.edge_index,edgeidxbb],1)
        gsbandbb.y=gsb.y
        datasetgh.append(gsbandbb)
    torch.save(datasetgh,f"GHT-online-single-{myname}-{opnamesim}.pt")
    #print('anykey to conitnue')
    #anykey=input()
    
    return None


#单位文件的在线数据测试
#在线时的分类概率测试
#nginter为间隔几局更新一个图
def testSGGDOnlineRecog(myname,opnamesim,ngstt=6,nginter=6,ngtrainlist=[3000]):

    dataset=torch.load(f"GHT-online-single-{myname}-{opnamesim}.pt")
    ndata=len(dataset)
    print('len(dataset)',len(dataset))
    recogRCGhis=torch.zeros((len(ngtrainlist),ndata,2)) #size为识别器数量*图数量*2,第三轴的两个数据是生成图的局数，识别为当前对手的概率
    j=0
    for ngtrainsamples in ngtrainlist:
        nnmodelfile=f'recog-op-GNN-Ngtr-{ngtrainsamples}.pkl'
        num_node_features=7
        hidden_channels=64
        num_graph_classes=14
        model =GINnet(num_node_features, hidden_channels, num_graph_classes)

        try:
            if os.path.exists(nnmodelfile):
                model.load_state_dict(torch.load(nnmodelfile,map_location=torch.device('cpu')))
        except FileNotFoundError:
            print('warning: parameter of the model was not loaded!')


        opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                    'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot'
                    ]
        opnameidx={'ElDonatoro':0,'Feste':1,'HITSZ':2,'Hugh_iro':3,'Hugh_tbr':4,'Intermission':5,'PokerBot5':6,
                    'PokerCNN':7,'PPPIMC':8,'Rembrant6':9,'RobotShark_iro':10,'RobotShark_tbr':11,'SimpleRule':12,'Slumbot':13}

        
        model.eval()
        recoghis=torch.zeros((ndata,2)) #size为图数量*2,第二轴的两个数据是生成图的局数，识别为当前对手的概率
        recogprobhis=torch.zeros((ndata,14)) #size为图数据*14，第二轴的14个数据为当前对手识别为所有对手的概率

        batchsize=1
        test_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        
        i=0
        for data in test_loader:                             # 批遍历测试集数据集。
            
            out = model(data.x, data.edge_index, data.batch) # 一次前向传播
            #print('out=',out)
            prob= F.softmax(out,dim=1)
            recogprobhis[i,:]=prob[0][:]
            #print('prob=',prob)
            yidx=data.y.item()
            #print('yidx=',yidx)
            classprob=prob[0][yidx].item()
            #print('prob[class]={:.6f}'.format(classprob))
            recoghis[i,0]=ngstt+i*nginter
            recoghis[i,1]=classprob
            recogRCGhis[j,i,:]=recoghis[i,:]
            i+=1
        print('recoghis=',recoghis,recoghis.size())


        #当前对手识别为各对手的概率
        plt.figure()
        for k in range(14):
            plt.plot(recoghis.numpy()[:,0],recogprobhis.detach().numpy()[:,k],
                        label="P of "+opnames[k],marker=markline[k],markevery=3,color=colors_use[k])
        plt.title(f'Recognition by RCG{ngtrainsamples} in match against {opnamesim}')
        plt.xlabel('number of games', fontsize=14)
        plt.ylabel('Porbability of each class',fontsize=14)
        plt.ylim(-0.05,1.05)
        #plt.xlim(0,min([ngtrainsamples*2,3000]))
        plt.legend(frameon=False,loc='center right',fontsize=10)
        plt.savefig(f'fig-op-recog-{opnamesim}-single-ngtr-{ngtrainsamples}.pdf')

        j+=1


    #当前对手识别为该对手的概率变化
    plt.figure()
    j=0
    for ngtrainsamples in ngtrainlist:
        plt.plot(recogRCGhis.numpy()[j,:,0],recogRCGhis.numpy()[j,:,1],
                    label=f"P by RCG{ngtrainsamples}",marker=markline[j],
                    markevery=3,color=colors_use[j])
        j+=1
    plt.title(f'Probability by different RCGs in match against {opnamesim}')
    plt.xlabel('number of games')
    plt.ylabel('Porbability')
    plt.ylim(-0.05,1.05)
    #plt.xlim(0,min([ngtrainsamples*2,500]))
    plt.legend(frameon=False,loc='center right',fontsize=8)
    plt.savefig(f'fig-op-recog-{opnamesim}-single-ngtr-diff-RCG.pdf')
    #plt.show()



    return None






if __name__ == "__main__":

    if 0:
        #######
        #在线阶段
        if 0:#数据生成
            preparedataonline(ngstt=6,ngend=3000,nginter=6)

        if 0:#测试
            testOnlineRecog(ngstt=6,nginter=6,ngtrainsamples=750,nettype='')

        if 1:
            testOnlineRecogAllclassifier(ngstt=6,nginter=6,ngtrainsampleslst=[50,100,300])
            testOnlineRecogAllclassifier(ngstt=6,nginter=6,ngtrainsampleslst=[500,750,1000],addfiglabel='B')
            pass
    
    if 0:
        #生成自定义数据集
        ftout=None
        testDATAsetcreate(3000,ftout)

    if 0:
        #
        #testTrainGraphClassifier(3000,'D',80,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='D')

        #testTrainGraphClassifier(3000,'E',100,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='E')

        #testTrainGraphClassifier(3000,'F',80,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='F')

        #testTrainGraphClassifier(3000,'G',80,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='G')

        #testTrainGraphClassifier(3000,'H',80,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='H')

        #testTrainGraphClassifier(3000,'I',80,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='I')

        #testTrainGraphClassifier(3000,'C',80,100)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='C')
        
        #testTrainGraphClassifier(3000,'B',100,100,0.0005)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='B')
        
        testTrainGraphClassifier(3000,'',80,100,0.0003)
        testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='')

        #testTrainGraphClassifier(3000,'A',40,100)
        #testGraphClassifier(ngtrainsamples=3000,ngtestsamples=3000,nettype='A')



    if 0:
        #######
        #训练阶段
        ftout=None
        if 0: #生成自定义数据集
            #ftout=False
            testDATAsetcreate(50,ftout)
            testDATAsetcreate(100,ftout)
            testDATAsetcreate(150,ftout)
            testDATAsetcreate(200,ftout)
            testDATAsetcreate(250,ftout)
            testDATAsetcreate(300,ftout)
            testDATAsetcreate(500,ftout)
            testDATAsetcreate(750,ftout)
            testDATAsetcreate(1000,ftout)
            testDATAsetcreate(3000,ftout)


        if 0:#训练并测试
            '''
            testTrainGraphClassifier(3000)
            testTrainGraphClassifier(1000)
            testTrainGraphClassifier(750)
            testTrainGraphClassifier(500)
            '''
            testTrainGraphClassifier(50)
            testTrainGraphClassifier(100)
            testTrainGraphClassifier(150)
            testTrainGraphClassifier(200)
            testTrainGraphClassifier(250)
            testTrainGraphClassifier(300)


        if 1:#测试结果
            resultpostdeal(ftout)


        if 0:#测试训练的模型
            testGraphClassifier(ngtrainsamples=3000,ngtestsamples=500)
    

    if 0:#单文件的绘图
        playername='TARD'
        opname='CI-PokerCNN_2pn_2017'
        #默认的log文件是小盲注先行的
        logfiledealing(f'Match-res-2p-{opname}-{playername}-200.log',playername,bigblindfirst=False)
        actiontreegen(0,3000)
        drawactiontree(GameTreeRecPoszero,myposSB=True)  #我方小盲注位时的博弈图
        drawactiontree(GameTreeRecPosones,myposSB=False) #我方大盲注位时的博弈图



    if 0:#单文件的识别测试
        playername='NN-ASHE_2pn_2017'
        #opnamesim='CI-PokerCNN_2pn_2017'
        opnamesim='NN-PokerCNN_2pn_2017'
        opnamereal='PokerCNN_2pn_2017'
        #filename="Match-res-2p-CI-PokerCNN_2pn_2017-NN-ASHE_2pn_2017-20000.log"
        filename="Match-res-2p-NN-PokerCNN_2pn_2017-NN-ASHE_2pn_2017-20000.log"
        #默认的log文件是小盲注先行的
        logfiledealing(filename,playername,bigblindfirst=False)
        actiontreegen(0,3000)
        drawactiontree(GameTreeRecPoszero,myposSB=True)  #我方小盲注位时的博弈图
        drawactiontree(GameTreeRecPosones,myposSB=False) #我方大盲注位时的博弈图

        
        #在线测试的图数据生成
        prepareSGGDonline(filename,playername,opnamesim,opnamereal,bigblindfirst=False,ngstt=50,ngend=3000,nginter=50,ftout=False)
        #在线测试
        testSGGDOnlineRecog(playername,opnamesim,ngstt=50,nginter=50,ngtrainlist=[50,100,150,200,300,500,750,1000,3000])


    if 1:
        #游戏局树的统计和聚类
        #从log得到的结果
        playername='ASHE'
        opname='ElDonatoro'
        #ACPC的log文件是大盲注先行的
        logfiledealing(f'Match-{playername}.{opname}-all.log',playername+'_2pn_2017',bigblindfirst=True)
       
        #print('press anykey to continue:')
        #anykey=input()
        if 0:
            actiontreegen(0,2)
            drawactiontree(GameTreeRecPoszero,myposSB=True,flgchain=True,addmark='orig2')  #我方小盲注位时的博弈图
            drawactiontree(GameTreeRecPosones,myposSB=False,flgchain=True,addmark='orig2') #我方大盲注位时的博弈图
            
            actiontreegen(0,10)
            drawactiontree(GameTreeRecPoszero,myposSB=True,flgchain=False,addmark='orig10')  #我方小盲注位时的博弈图
            drawactiontree(GameTreeRecPosones,myposSB=False,flgchain=False,addmark='orig10') #我方大盲注位时的博弈图
        
            actiontreegen(0,3000)
            drawactiontree(GameTreeRecPoszero,myposSB=True,flgchain=False,addmark='orig3000')  #我方小盲注位时的博弈图
            drawactiontree(GameTreeRecPosones,myposSB=False,flgchain=False,addmark='orig3000') #我方大盲注位时的博弈图

        if 1:
            actiontreegen(0,500)
            drawactiontree(GameTreeRecPoszero,myposSB=True,flgchain=False,addmark='orig500')  #我方小盲注位时的博弈图
            

        '''
        for data in recAllresults[:10]:
            print(data['msgsbfirst'])
        '''

        #tree1=actiontreecluster()
        #print('press anykey to continue:')
        #anykey=input()
        #tree2=actiontreecluster()
        #comparehistree(tree2,tree1)


    
    plt.show()

    pass



    

    

    
    



