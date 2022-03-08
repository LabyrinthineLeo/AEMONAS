import numpy as np
from archit_entropy import archit_entropy_value

# @Author : Labyrinthine Leo
# @Time   : 2021.05.20
# @fun    : naga2 public function: Nondominated-Sort、Crowd-Distance、Mating、EnvironmentSelection

def sortrows(Matrix, order="ascend"):

    Matrix_temp = Matrix[:, ::-1]
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank, :]

    return Sorted_Matrix, rank

def NDSort(PopObj, Remain_Num):
    N, M = PopObj.shape
    FrontNO = np.ones((1, N))
    PopObj, rank = sortrows(PopObj)

    for i in range(Remain_Num):
        obj_1 = PopObj[i][0]
        obj_2 = PopObj[i][1]
        obj_x = FrontNO[0][i]
        for j in range(i+1, Remain_Num):
            if (PopObj[j][0] == obj_1 and PopObj[j][1] == obj_2) or (PopObj[j][0] > obj_1 and PopObj[j][1] < obj_2):
                continue
            elif FrontNO[0][j] > obj_x:
                continue
            else:
                FrontNO[0][j] += 1


    front_temp = -np.ones((1, N))
    # print(FrontNO)
    front_temp[0, rank[:Remain_Num]] = FrontNO[0, :Remain_Num]

    return front_temp, int(np.max(FrontNO))

def Distance(ObjValue, FrontValue):

    N, M = ObjValue.shape
    CrowdDistance = np.zeros((1, N))
    MaxFrontNo = int(np.max(FrontValue))
    for i in range(MaxFrontNo):
        is_ = np.array(FrontValue[0] == i+1)
        ind_id = np.where(FrontValue[0] == i + 1)[0]
        y = ObjValue[is_, :]
        y = np.c_[y, ind_id].tolist()
        y.sort(key=lambda x: (-x[0], x[1]))
        y = np.array(y)

        obj1_range = y[0][0] - y[-1][0]
        obj2_range = y[-1][1] - y[0][1]
        crowd_list = []
        for i in range(y.shape[0]):
            if i == 0 or i == y.shape[0]-1:
                crowd_list.append(np.inf)
            else:
                crowd = np.abs(y[i+1][0]-y[i-1][0])/obj1_range + np.abs(y[i+1][1]-y[i-1][1])/obj2_range
                crowd_list.append(crowd)
        CrowdDistance[0, y[:, 2].astype(int)] = np.array(crowd_list)

    return CrowdDistance #

def Mating(Population, ObjValue, FrontValue, CrowdDistance):

    N = len(Population) #
    MatingPool = [] #
    MatingPool_index = [] #
    Rank = np.random.permutation(N) #
    Pointer=0 #
    for i in range(0, N, 2):
        k = [0, 0]
        for j in range(2):
            if Pointer+1 >= N:
                Rank = np.random.permutation(N)
                Pointer = 0

            p = Rank[Pointer]
            q = Rank[Pointer+1]
            if FrontValue[0, p] < FrontValue[0, q]:
                k[j] = p
            elif FrontValue[0, p] > FrontValue[0, q]:
                k[j] = q
            elif CrowdDistance[0, p] == np.inf or CrowdDistance[0, q] == np.inf:
                if ObjValue[p, 0] < ObjValue[q, 0]:
                    k[j] = p
                else:
                    k[j] = q
            elif CrowdDistance[0, p] > CrowdDistance[0, q]:
                k[j] = p
            else:
                k[j] = q

            Pointer += 2

        MatingPool_index.extend(k[0:2])
        MatingPool.append(Population[k[0]])
        MatingPool.append(Population[k[1]])

    return MatingPool, MatingPool_index

# ==================== Crossover and Mutation ====================
def links_mutation(links_encode):
    zero_index = (links_encode == 0)
    one_index = (links_encode == 1)
    links_encode[zero_index] = 1
    links_encode[one_index] = 0

    return links_encode

def ops_mutation(ops_encode):
    ops_dict = {}
    for i in range(12):
        ops_dict[i] = 0
    op_nums = 11
    for op in ops_encode:
        ops_dict[op] += 1
    prob = [(ops_dict[i] + 1) / (op_nums + len(ops_encode)) for i in range(op_nums)]
    new_ops = []
    for op in ops_encode:
        new_op = np.random.choice(op_nums, 1, p=prob)[0]
        while new_op == op:
            new_op = np.random.choice(op_nums, 1, p=prob)[0]
        new_ops.append(new_op)

    return np.array(new_ops)

def rectify(cell_dag):
    link_dag = cell_dag[0] # links
    op_dag = cell_dag[1] # ops
    begin_id = 0 # begin index
    node_id = 0 # index of node
    zero_index = [] # index of resetting the zero
    temp_encodes = [[], []]
    while begin_id < len(link_dag):
        begin = begin_id
        begin_id += 2 + node_id
        node_encode = np.array(link_dag[begin:begin_id]).copy()
        if node_encode.sum() - node_encode[zero_index].sum() == 0: # only the link that would be deleted is 1
            zero_index.extend([node_id+2])
            node_encode[-1] = 1
        temp_encodes[0].extend(node_encode)
        node_id += 1

    temp_encodes[1] = op_dag
    return temp_encodes.copy()

def Generator(MatingPool, Coding, MaxOffspring):

    N = len(MatingPool)
    MatingPool_temp = MatingPool.copy()
    MatingPool_encodes = [i.encodes for i in MatingPool]
    ori_AE = archit_entropy_value(MatingPool_encodes)

    if MaxOffspring < 1 or MaxOffspring > N:
       MaxOffspring = N

    # assert Coding != "Binary" 'The encoding should be binary!'

    Offspring = []
    cross_ratio = 0.4  # 0.2

    if Coding == "Binary":
        for idx in range(5):
            for i in range(0, N, 2):
                P1 = MatingPool_temp[i].encodes.copy()
                P2 = MatingPool_temp[i+1].encodes.copy()

                cross_flag = np.random.rand(1) < cross_ratio

                for j in range(2):
                    p1_links = np.array(P1[j][0]).copy()
                    p1_ops = np.array(P1[j][1]).copy()
                    p2_links = np.array(P2[j][0]).copy()
                    p2_ops = np.array(P2[j][1]).copy()

                    p1_links_len = p1_links.shape[0]
                    p2_links_len = p2_links.shape[0]
                    p1_ops_len = p1_ops.shape[0]
                    p2_ops_len = p2_ops.shape[0]
                    len_flag = p1_links_len > p2_links_len
                    links_common_len =  p2_links_len if len_flag else p1_links_len
                    links_cross_point = np.random.choice(links_common_len)
                    ops_common_len = p2_ops_len if len_flag else p1_ops_len
                    ops_cross_point = np.random.choice(ops_common_len)

                    if cross_flag:
                        p1_links[:links_cross_point], p2_links[:links_cross_point] = p2_links[:links_cross_point], p1_links[:links_cross_point]
                        p1_ops[:ops_cross_point], p2_ops[:ops_cross_point] = p2_ops[:ops_cross_point], p1_ops[:ops_cross_point]

                    links_muta_flag_1 = np.random.rand(len(p1_links), ) < 3 / len(p1_links)
                    links_muta_flag_2 = np.random.rand(len(p2_links), ) < 3 / len(p2_links)
                    links_muta_1 = links_mutation(p1_links.copy())
                    links_muta_2 = links_mutation(p2_links.copy())
                    ops_muta_flag_1 = np.random.rand(len(p1_ops), ) < 1 / len(p1_ops)
                    ops_muta_flag_2 = np.random.rand(len(p2_ops), ) < 1 / len(p2_ops)
                    ops_muta_1 = ops_mutation(p1_ops.copy())
                    ops_muta_2 = ops_mutation(p2_ops.copy())

                    p1_links[links_muta_flag_1] = links_muta_1[links_muta_flag_1]
                    p1_ops[ops_muta_flag_1] = ops_muta_1[ops_muta_flag_1]
                    p2_links[links_muta_flag_2] = links_muta_2[links_muta_flag_2]
                    p2_ops[ops_muta_flag_2] = ops_muta_2[ops_muta_flag_2]

                    P1[j] = rectify([list(p1_links.copy()), list(p1_ops.copy())])
                    P2[j] = rectify([list(p2_links.copy()), list(p2_ops.copy())])

                if not cross_flag:
                    temp_p1 = P1.copy()
                    P1[1] = P2[1]
                    P2[1] = temp_p1[1]


                Offspring.append(P1)
                Offspring.append(P2)


            np.random.shuffle(MatingPool_temp)


    ae_list = [] #
    for idx in Offspring:
        ae_list.append(archit_entropy_value([*MatingPool_encodes, idx]) - ori_AE)

    ae_rank = np.argsort(-np.array(ae_list))
    Offspring_ = np.array(Offspring)[ae_rank[:MaxOffspring]].tolist()

    return Offspring_

def EnvironmentSelect(Population, FunctionValue, N):
    FrontValue, MaxFront = NDSort(FunctionValue, N)
    select_index = [i for i, v in enumerate(FrontValue[0]) if v != -1]

    new_Population = [Population[i] for i in select_index]
    new_FunctionValue = np.array([FunctionValue[i] for i in select_index])
    new_FrontValue = np.array([[FrontValue[0][i] for i in select_index]])

    CrowdDistance = Distance(new_FunctionValue, new_FrontValue)

    return new_Population, new_FunctionValue, new_FrontValue, CrowdDistance, select_index
