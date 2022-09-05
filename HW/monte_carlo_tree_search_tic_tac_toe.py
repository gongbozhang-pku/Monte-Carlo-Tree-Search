from abc import ABC, abstractmethod
from collections import defaultdict  # 用来生成默认值 list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
import numpy as np
from numpy import  log, sqrt


class MCTS:
                                                       # add
    def __init__(self, exploration_weight=5000, policy='uct', budget=1000, n0=10, opp_policy='random', sigma_0=1):
        self.Q = defaultdict(float)  # total reward of each node 存放每个节点的总回报
        self.V_bar = defaultdict(float) # V_bar、V_hat都是后续需要的参数，该定义返回值 0.0
        self.V_hat = defaultdict(float)
        self.N = defaultdict(int)  # total visit count for each node 存放每个节点总访问次数
        self.children = defaultdict(set)  # children of each node 每个节点的子节点
        self.exploration_weight = exploration_weight # 探索某节点的概率
        self.all_Q = defaultdict(list)  # total reward of all nodes
        assert policy in {'uct', 'ocba'}, 'Policy must be either uct or ocba!' 
        self.policy = policy
        self.std = defaultdict(float)  # std of each node 各节点标准差
        self.ave_Q = defaultdict(float) # 存放每个节点的平均回报
        self.budget=budget #最大访问次数
        self.n0 = n0
        self.leaf_cnt = defaultdict(int) #存放叶子节点数量
        self.opp_policy = opp_policy
        self.sigma_0 = sigma_0
        

    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child() #从子节点中随机找一个

        def score(n):
            if self.N[n] == 0: # 某节点的访问次数为0
                return float("-inf")  # avoid unseen moves #则返回负无穷
            return self.ave_Q[n]  # average reward 不为零则返回平均回报
        rtn = max(self.children[node], key=score) # 选出某节点的子节点中分数最高的
        
        return rtn

    def do_rollout(self, node):
        
        path = self._select(node)
        leaf = path[-1]
        
        self.leaf_cnt[leaf] += 1 #叶子节点数加一
        sim_reward = self._simulate(leaf)
        self._backpropagate(path, sim_reward)

    def _select(self, node):
        
        path = []
        while True:
            path.append(node) # path中加入未探索的节点
            self.N[node] += 1 #该节点的访问次数+1
            if node.terminal:#判断该节点是否为terminal状态
                # node is either unexplored or terminal
                return path
            
            if len(self.children[node]) == 0:# 第一次访问节点是 其子节点数量为0
                # First time to visit a state node, add its children to self.children
                children_found = node.find_children()
                self.children[node] = children_found
                
            
            if node.turn == -1:# 对手下棋
                # opponent's turn
                if self.opp_policy == 'random': #若其policy为random
                    node = node.find_random_child() #则随机选择子节点
                elif self.opp_policy == 'uct':  #若其policy为uct 下面为uct算法
                    expandable = [n for n in self.children[node] if self.N[n] < 1] #expand当前没有visit过的节点
                    if expandable: #若存在没有visit过的节点
                        node = expandable.pop() #则使其为当前节点 并从expandable（可以理解为待探索节点集）中去除
                    else:#若不存在
                        log_N_vertex = log(sum([self.N[c] for c in self.children[node]])) # 子节点被访问的次数之和的log
                        node = min(self.children[node], key=lambda n:self.ave_Q[n] #对手的uct为最小化报酬即树的置信下界
                                   - self.exploration_weight * sqrt( 2 * log_N_vertex / self.N[n]))
                continue
            
            expandable = [n for n in self.children[node] if self.N[n] < self.n0] #将访问次数小于n0的节点加入expand
            # 自己下棋
            if  expandable:#自己的策略uct/ocba总是选择expand
                # expandable
                a = self._expand(node)
                if len(self.children[a]) == 0:#若无子节点则选择子节点
                    self.children[a] = a.find_children()
                path.append(a)
                self.N[a] += 1  #该节点的访问次数+1
                
                return path
            else: #若没有待探索的节点，则根据策略不同选择不同方法
                if self.policy == 'uct':
                    a = self._uct_select(node)  # descend a layer deeper
                else:
                    a = self._ocba_select(node)
                node = a

    def _expand(self, node, path_reward=None):
        
        explored_once = [n for n in self.children[node] if self.N[n] < self.n0] #将访问次数小于n0的节点加入expand_once
        return explored_once.pop() #取出最后一个并将其从explored_once中删除

    def _simulate(self, node):
        
        while True:
            if not node.is_terminal(): #若节点不为终点，则发展子节点
                node = node.find_random_child()
            if node.terminal: #若节点为终点，则返回reward
                return node.reward()

    def _backpropagate(self, path, r):
        "Send the reward back up to the ancestors of the leaf"
        for i in range(len(path)-1, -1, -1):
            node = path[i] #从后往前逐个读取节点，叶子节点不读
            '''
            Iteratively update std, which is supposed to be faster.
            Population std.
            '''
            self.Q[node] += r #更新该节点的回报
            self.all_Q[node].append(r)
            
            old_ave_Q = self.ave_Q[node] #保存之前的平均回报
            self.ave_Q[node] = self.Q[node] / self.N[node] #更新当前平均回报
            # 计算节点的标准差
            self.std[node] = sqrt(((self.N[node]-1)*self.std[node]**2 + (r - old_ave_Q) * (r - self.ave_Q[node]))/self.N[node]) 
            
            if self.std[node] == 0:
               self.std[node] = self.sigma_0
              

    def _uct_select(self, node):
        
        # all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False
        # 元素除了是 0、空、None、False 外都算 True ##为false时触发assert
        assert all(n in self.children for n in self.children[node])  #确保子节点不为空

        log_N_vertex = log(sum([self.N[c] for c in self.children[node]])) #

        def uct(n):
            "Upper confidence bound for trees 树的置信上界"
            return self.ave_Q[n] + self.exploration_weight * sqrt(
                2 * log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct) #返回某节点下uct最大的子节点
    
    def _ocba_select(self, node):
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node])>0, "Error! Empty children action set!"
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0] #若只有一个子节点，则读取它
        
        all_actions = self.children[node] #将所有子节点存到all_actions里
        b = max(all_actions, key=lambda n: self.ave_Q[n]) #选平均回报最高的
        best_Q = self.ave_Q[b] #存到best_Q
        suboptimals_set, best_actions_set, select_actions_set = set(), set(), set() #定义空集合存储次优集和最优行动集
        for k in all_actions:
            if self.ave_Q[k] == best_Q:
                best_actions_set.add(k) #平均回报最高的子节点被存入最优行动集 其余节点在次优集
            else:
                suboptimals_set.add(k)
        
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n]) #若次优集为空，则选择遍历次数最小的子节点
        
        if len(best_actions_set) != 1:
            b = max(best_actions_set, key=lambda n : (self.std[node])**2 / self.N[n]) #若最优集包含不止一个，则选择遍历次数最小方差最大的子节点为最优
            
        for k in all_actions:
            if self.ave_Q[k] != best_Q:
                select_actions_set.add(k)
        select_actions_set.add(b)
        
        delta = defaultdict(float) #定义delta
        for k in select_actions_set:
            delta[k] = self.ave_Q[k] - best_Q # Δ =子节点各自的平均回报值-最优节点的平均回报值
        
        # Choose a random one as reference
        ref = next(iter(suboptimals_set)) #从次优集中选择一个

        para = defaultdict(float)
        ref_std_delta = self.std[ref]/delta[ref] #std_delta=std/delta
        para_sum = 0
        for k in suboptimals_set:
            para[k] = ((self.std[k]/delta[k])/(ref_std_delta))**2 #ref的para为1，其他为相对值 (std/delta//std/delta)²
               
        para[b] = sqrt(sum((self.std[b]*para[c]/self.std[c])**2 for c in suboptimals_set))

        para_sum = sum(para.values()) #values()以列表返回字典中的所有值  该式对字典的所有值求和
        para[ref] = 1
       
        totalBudget = sum([self.N[c] for c in select_actions_set])+1
        ref_sol = (totalBudget)/para_sum
        
        return max(select_actions_set, key=lambda n:para[n]*ref_sol - self.N[n])

    def _AOAP_select():
      #请在该函数下补充完整
    def _TTTS_select():
        #请在该函数下补充完整
       

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
