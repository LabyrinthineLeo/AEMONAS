import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import Operations_11, Operations_name, FactorizedReduce,MaybeCalibrateSize, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path
import numpy as np

# @Author : Labyrinthine Leo
# @Time   : 2020.12.29

class Node(nn.Module):
    def __init__(self, previous_shape, dag_node, channels, cell_id, total_layers,stride=1,
                 drop_path_keep_prob=None, steps=0):
        super(Node,self).__init__()

        self.drop_path_keep_prob = drop_path_keep_prob
        self.steps = steps
        self.cell_id = cell_id
        self.total_layers = total_layers

        self.adj_node = dag_node.adj_node
        self.node_id = dag_node.node_id
        self.node_name = Operations_name[dag_node.op_id]
        # print(self.node_name)
        self.Operation = Operations_11
        self.previous_shape = previous_shape
        self.linked_previous = [x for i, x in enumerate(previous_shape) if self.adj_node[i]==1]

        self.channels = channels
        self.stride = stride

        assert previous_shape[0] == previous_shape[1]
        self.out_shape = [previous_shape[0][0]//stride, previous_shape[0][1]//stride, channels]

        self.reduction_flag = True if stride == 2 else False
        self.Factor_flag = any(self.adj_node[2:])

        stride = 1 if self.Factor_flag else stride

        self.ops = self.Operation[dag_node.op_id](channels, channels, stride, self.out_shape, True)

        # self.alpha = torch.nn.Parameter(torch.FloatTensor(6))

        # self.identify = FactorizedReduce(channels, channels, self.out_shape, True)

    def forward(self, x_input, step):
        # deal with the situation that the node in Reduction Cell
        # connects to these nodes having different size feature maps
        if self.reduction_flag: # 是reduction cell
            if self.Factor_flag or self.node_name in ['Identity', 'SELayer']:
                x_input.pop(0)
                x_input.pop(0)
            else:
                x_input.pop(2)
                x_input.pop(2)
        # add different input, when only one input, the sum operation is also correct
        x = sum([x for i, x in enumerate(x_input) if self.adj_node[i]==1])
        # via operation
        x = self.ops(x)

        if self.drop_path_keep_prob is not None and self.training:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.cell_id, self.total_layers, step, self.steps)

        return x

class Cell(nn.Module):
    def __init__(self, dag, prev_layers, channels, reduction, cell_id, total_layers,
                 drop_path_keep_prob=None, steps=0):

        super(Cell, self).__init__()

        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.total_layers = total_layers
        self.cell_id = cell_id


        self.Factor_flag = False #
        self.dag = dag #

        self.prev_layers = prev_layers # [layer1_shape, layer2_shape]
        self.reduction = reduction # flag of reduction
        self.channels = channels # channel

        self.num_node = len(dag) - 2

        self.stride = 2 if self.reduction else 1

        self.ops = nn.ModuleList()
        # maybe calibrate size
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        Iterated_prev_layers = self.maybe_calibrate_size.out_shape

        self.used = np.array([0] * (self.num_node + 2))

        for i in range(self.num_node):
            node = Node(Iterated_prev_layers, dag[i], channels,
                        cell_id=self.cell_id, total_layers=self.total_layers,
                        stride=self.stride,
                        drop_path_keep_prob=self.drop_path_keep_prob,
                        steps=self.steps)

            self.ops.append(node)

            Iterated_prev_layers.append(node.out_shape)

            self.used[np.where(np.array(dag[i].adj_node)==1)[0]] = 1


        self.shape_for_node = Iterated_prev_layers
        self.concat = [i for i in range(self.num_node+2) if self.used[i]==0]

        if self.reduction:
            self.fac_1 = FactorizedReduce(self.shape_for_node[0][-1], channels, self.shape_for_node[0])
            self.fac_2 = FactorizedReduce(self.shape_for_node[1][-1], channels, self.shape_for_node[1])


        out_hw = min([shape[0] for i, shape in enumerate(self.shape_for_node) if i in self.concat])

        self.out_shape = [out_hw, out_hw, channels * len(self.concat)]

    def forward(self, s0, s1, step, bn_train=False):
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]

        if self.reduction:
            states.append(self.fac_1(s0, bn_train=bn_train))
            states.append(self.fac_2(s1, bn_train=bn_train))

        for i in range(self.num_node):
            out = self.ops[i](states.copy(), step)
            states.append(out)

        if self.reduction:
            states.pop(0)
            states.pop(0)

        out = torch.cat([states[i] for i in self.concat], dim=1)

        return out

class NetworkSpine(nn.Module):
    def __init__(self, args, classes, layers, channels,
                 dag, use_aux_head, keep_prob, steps, drop_path_keep_prob, channels_double=True):

        super(NetworkSpine,self).__init__()
        self.args = args
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.channels_double = channels_double
        self.use_aux_head = use_aux_head

        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.steps = steps

        self.dag = dag
        self.conv_dag = dag[0] # normal cell dag, dict type
        self.reduc_dag = dag[1] # reduction cell dag, dict type

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.total_layers = self.layers * 3 + 2

        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels//2, 3, stride=2, padding=1, bias=False),
            # output: 128*8*112*112
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels, 3, stride=2, padding=1, bias=False),
            # output: 128*16*56*56
            nn.BatchNorm2d(channels),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            # output: 128*16*28*28
            nn.BatchNorm2d(channels),
        )

        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels

        self.cells = nn.ModuleList() # the list of store modules
        for i in range(self.total_layers):
            if i not in self.pool_layers: # normal cell
                cell = Cell(self.conv_dag, prev_layers=outs, channels=channels,
                            reduction=False, cell_id=i, total_layers=self.total_layers,
                            drop_path_keep_prob=self.drop_path_keep_prob,
                            steps=self.steps)
            else:
                if self.channels_double:
                    channels *= 2 # double the channel
                cell = Cell(self.reduc_dag, prev_layers=outs, channels=channels,
                            reduction=True, cell_id=i, total_layers = self.total_layers,
                            drop_path_keep_prob = self.drop_path_keep_prob,
                            steps=self.steps)
            self.cells.append(cell)

            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data) #

    def forward(self, input, step=None, bn_train=False):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)

        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.use_aux_head:
            return logits, aux_logits
        else:
            return logits

class NetworkCIFAR(nn.Module):
    def __init__(self, args, classes, layers, channels,
                 dag, use_aux_head, keep_prob,steps,drop_path_keep_prob, channels_double=False):

        super(NetworkCIFAR,self).__init__()
        self.args = args
        self.classes = classes #
        self.layers = layers  # each Normal Cell repeat times
        self.channels = channels #
        self.channels_double = channels_double # used for double channels after Reduction Cell, True for ImageNet or bigger model
        self.use_aux_head = use_aux_head # head

        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.steps = steps

        self.dag = dag
        self.conv_dag = dag[0] # normal cell dag
        self.reduc_dag = dag[1] # reduction cell dag

        self.pool_layers = [self.layers, 2 * self.layers + 1] # reduc cell id
        self.total_layers = self.layers*3 +2  # 3*N + 2

        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        stem_multiplier = 3 #
        channels = stem_multiplier * self.channels #

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False), # in_channel:3, out_channels:channels
            nn.BatchNorm2d(channels)
        )

        outs = [[32, 32, channels], [32, 32, channels]]

        channels = self.channels #
        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers: # norm cell
                cell = Cell(self.conv_dag, prev_layers=outs, channels=channels, #
                            reduction=False, cell_id=i, total_layers = self.total_layers,
                            drop_path_keep_prob = self.drop_path_keep_prob,
                            steps=self.steps)
            else:
                if self.channels_double:
                    channels *= 2
                cell = Cell(self.reduc_dag, prev_layers=outs, channels=channels,
                            reduction=True, cell_id=i, total_layers = self.total_layers,
                            drop_path_keep_prob = self.drop_path_keep_prob,
                            steps=self.steps)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes) #

        self.init_parameters() #

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, step=None, bn_train=False):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step, bn_train= bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.use_aux_head:
            return logits, aux_logits
        else:
            return logits

class NetworkImageNet(nn.Module):
    def __init__(self, args, classes, layers, channels,
                 dag, use_aux_head, keep_prob, steps, drop_path_keep_prob, channels_double=True):
        super(NetworkImageNet,self).__init__()
        self.args = args
        self.classes = classes
        self.layers = layers  # each Normal Cell repeat times
        self.channels = channels
        self.channels_double = channels_double # used for double channels after Reduction Cell, True for ImageNet or bigger model
        self.use_aux_head = use_aux_head

        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.steps = steps

        self.dag = dag
        self.conv_dag = dag[0]
        self.reduc_dag = dag[1]

        self.pool_layers = [self.layers, 2 * self.layers + 1] #
        self.total_layers = self.layers*3 + 2

        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels

        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(self.conv_dag, prev_layers=outs, channels=channels,
                            reduction=False, cell_id=i,total_layers = self.total_layers,
                            drop_path_keep_prob = self.drop_path_keep_prob,
                            steps=self.steps)
            else:
                if self.channels_double:
                    channels *= 2
                cell = Cell(self.reduc_dag, prev_layers=outs, channels=channels,
                            reduction=True, cell_id=i, total_layers = self.total_layers,
                            drop_path_keep_prob = self.drop_path_keep_prob,
                            steps=self.steps)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, step=None, bn_train=False):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step, bn_train= bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.use_aux_head:
            return logits, aux_logits
        else:
            return logits




if __name__ == '__main__':
    pass
