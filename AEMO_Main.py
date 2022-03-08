import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import collections, argparse, time, logging, sys
import matplotlib.pyplot as plt

from nsga_public import NDSort, Distance, Mating, Generator, EnvironmentSelect

from model_training import solution_evaluation
from utils import dagnode, create_dir, Plot_network
from operations import Operations_name
from cell_archit import NetworkSpine, NetworkCIFAR, NetworkImageNet
from build_dataset import get_cifar10_dataloader, get_cifar100_dataloader, build_search_spine3, build_search_Optimizer_Loss

# @Author : Labyrinthine Leo
# @Time   : 2021.01.01

# Individual Class
class Individual():
    def __init__(self, encodes):
        """
        init function
        :param encodes: normal cell and reduction cell encodes
        """
        self.encodes = encodes
        print(self.encodes)
        self.rectify()
        self.trans2dag()

    def rectify(self):
        # used for deleting the nodes not actived(all 0)
        for i, cell_dag in enumerate(self.encodes):
            link_dag = cell_dag[0] # links
            op_dag = cell_dag[1] # ops
            begin_id = 0 # begin index
            node_id = 0 # index of node
            zero_index = [] # index of resetting the zero
            temp_encodes = [[], []]
            # 删除失活节点
            while begin_id < len(link_dag):
                begin = begin_id
                begin_id += 2 + node_id
                node_encode = np.array(link_dag[begin:begin_id]).copy()
                if node_encode.sum() - node_encode[zero_index].sum() == 0: # only the link that would be deleted is 1
                    zero_index.extend([node_id+2])
                    node_encode[-1] = 1 # activate the last node
                # else:
                #     temp_encodes[0].extend(np.delete(node_encode, zero_index)) # delete the 0 link
                #     temp_encodes[1].extend([op_dag[node_id]]) # append operation
                temp_encodes[0].extend(node_encode)
                node_id += 1

            temp_encodes[1] = op_dag # original operation
            self.encodes[i] = temp_encodes.copy()

    def trans2dag(self):
        """
        transform encodes to dag
        :return: None
        """
        self.dag = []
        self.num_node = [] # node nums of norm cell and reduc cell

        for i, cell_dag in enumerate(self.encodes):
            link_dag = cell_dag[0]  # links
            op_dag = cell_dag[1]  # ops

            dag = collections.defaultdict(list) # dict type, default value type: list
            dag[-2] = dagnode(-2, [], None)
            dag[-1] = dagnode(-1, [0], None)

            begin_id = 0
            node_id = 0
            while begin_id < len(link_dag): # iterate the link list
                begin = begin_id
                begin_id += 2 + node_id
                node_encode = link_dag[begin:begin_id]
                node_op = op_dag[node_id]
                dag[node_id] = dagnode(node_id, node_encode, node_op)
                node_id += 1

            self.num_node.extend([node_id])
            self.dag.append(dag)
            del dag

    def evaluate(self, args, train_queue, valid_queue):
        # evaluate the fitness of individual

        # fitness
        self.fitness = np.random.rand(4, )

        print(self.encodes)
        # logging.info(self.encodes)

        # get the type of datasets
        type_data = args.dataset

        if type_data == 'spine3':
            model = NetworkSpine(args, 3, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                                    args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                                    args.search_channels_double)
        elif type_data == 'cifar10':
            print("building cifar net.")
            # model = NetworkCIFAR(args, 10, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
            #                         args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
            #                         args.search_channels_double)
            model = NetworkCIFAR(args, 10, 1, 16, self.dag, False,
                                 args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                                 args.search_channels_double)
        elif type_data == 'cifar100':
            model = NetworkCIFAR(args, 100, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                                    args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                                    args.search_channels_double)
        elif type_data == 'ImageNet':
            model = NetworkImageNet(args, 1000, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                                    args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                                    args.search_channels_double)
        else:
            model = NetworkSpine(args, 3, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                                 args.search_keep_prob, args.search_steps, args.search_drop_path_keep_prob,
                                 args.search_channels_double)


        self.fitness = solution_evaluation(model, train_queue, valid_queue, args)

        del model


# Multi-Obj Evo Process Class
class MOEA():
    def __init__(self, args, visualization=False):
        """
        initialize the class
        :param args: arguments
        :param visualization: flag of visual
        """
        self.args = args # command arguments
        self.popsize = args.popsize # population size
        self.Max_Gen = args.Max_Gen # max generation
        self.Gen = 0 # current generation id
        self.initial_range_node = args.range_node # range of nodes
        self.save_dir = args.save # the dir of result saves

        self.op_nums = len(Operations_name) # operation nums
        self.coding = 'Binary'

        self.visualization = visualization

        self.Population = [] # population of cell structure
        self.Pop_fitness = [] # fitness of individual on pops
        self.fitness_best = 0 # the best fitness

        self.offspring = [] # offspring
        self.off_fitness = [] # fitness of offspring

        self.tour_index = [] # tournament selection index
        self.FrontValue = [] # pareto front
        self.CrowdDistance = [] # crowd distance

        if args.dataset == 'spine3':
            self.build_spine_dataset() # get spine dataset
        elif args.dataset == 'cifar10':
            print('build cifar10 dataset')
            self.build_cifar10_dataset() # get cifar dataset
        elif args.dataset == 'cifar100':
            self.build_cifar100_dataset() # get cifar dataset
        else:
            self.build_imagenet_dataset() # get imagenet dataset

        self.threshold = 0.08 # threshold

    def build_spine_dataset(self):
        """
        Building the spine dataset(3 classes), and get the train/valid/test queue
        :return: None
        """
        train_queue, valid_queue, test_queue = build_search_spine3(root_path=args.data, batch_size=args.search_train_batch_size, num_workers=self.args.search_num_work)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def build_cifar10_dataset(self):
        """
        Building the cifar dataset(10/100 classes), and get the train/valid queue
        :return: None
        """
        train_queue, valid_queue, test_queue = get_cifar10_dataloader(batch_size=self.args.search_train_batch_size, num_workers=self.args.search_num_work, shuffle=False)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def build_cifar100_dataset(self):
        """
        Building the cifar dataset(10/100 classes), and get the train/valid queue
        :return: None
        """
        train_queue, valid_queue, test_queue = get_cifar100_dataloader(batch_size=self.args.search_train_batch_size, num_workers=self.args.search_num_work, shuffle=False)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def build_imagenet_dataset(self):
        """
        Building the imagenet dataset(1000 classes), and get the train/valid queue
        :return: None
        """
        train_queue, valid_queue, test_queue = get_cifar100_dataloader(batch_size=self.args.search_train_batch_size, num_workers=self.args.search_num_work, shuffle=False)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def initialization(self):
        # initialize the population
        for pop in range(self.popsize):
            # used for controlling the network structure between line and inception
            rate = (pop+1) / self.popsize
            # get random nums of normal cell node and reduction cell node
            node_ = np.random.randint(self.initial_range_node[0], self.initial_range_node[1], 2)
            node_ = []
            # the nums of normal node
            # node_.extend(np.random.randint(self.initial_range_node[0], self.initial_range_node[1]+1, 1))
            # the nums of reduction node
            # node_.extend(np.random.randint(self.initial_range_node[0], self.initial_range_node[1]+1, 1))

            # print(node_)

            list_individual = [] # individual list

            for i, num in enumerate(node_):
                op = np.random.randint(0, self.op_nums, num) # random init operation
                if i == 0: # normal cell conv ops
                    op_c = np.random.choice([1, 2, 3, 4, 5, 10], num)
                else: # reduction cell pooling ops
                    op_c = np.random.choice([0, 1, 6, 7, 8, 9], num)

                indicator = np.random.rand(num) < 0.8 # 0.8 # setting changing threshold=0.8
                op[indicator] = op_c[indicator] # get new ops

                L = 2
                cell_dag_encode = [[], []] # cell-based dag encode, link and ops
                for j in range(num):
                    L += 1
                    link = np.random.rand(L-1) # random init link
                    link[-1] = link[-1] > rate
                    link[0:2] = link[0:2] < rate
                    link[2:-1] = link[2:-1] < 2 / len(link[2:-1]) if len(link[2:-1]) != 0 else []

                    if link.sum() == 0: # when all links are 0, adjusting them.
                        if rate < 0.5:
                            link[-1] = 1
                        else:
                            if np.random.rand(1) < 0.5:
                                link[1] = 1
                            else:
                                link[0] = 1

                    link = np.int64(link) # float to int
                    link = link.tolist() # ndarray to list
                    cell_dag_encode[0].extend(link) # append the link encode
                    cell_dag_encode[1].extend([op[j]]) # append the ops encode

                list_individual.append(cell_dag_encode)


            self.Population.append(Individual(list_individual))

        # Up_boundary = np.ones((self.max_length))
        # Up_boundary[self.op_index] = 11
        # Low_boundary = np.zeros((self.max_length))
        # self.Boundary = np.vstack((Up_boundary, Low_boundary))

        self.all_fitness = self.evaluation(self.Population)
        # error and Params
        self.Pop_fitness = self.all_fitness[:, :2]
        # error and FlOPs
        self.EF_fitness = self.all_fitness[:, [0, 2]]

        self.fitness_best = np.min(self.Pop_fitness[:, 0])
        self.save('initial')

    def save(self, path=None):
        """
        :param path:
        :return:
        """

        if path is None:
            path = 'Gene_{}'.format(self.Gen+1)
        whole_path ='{}/{}/'.format(self.save_dir, path)
        create_dir(whole_path)

        # fitness
        err_fitness_file = whole_path + 'err_param_fitness.txt'
        np.savetxt(err_fitness_file, self.Pop_fitness, delimiter=' ')
        #
        all_fitness_file = whole_path + 'all_fitness.txt'
        np.savetxt(all_fitness_file, self.all_fitness, delimiter=' ')


        Pop_file = whole_path + 'Population.txt'
        with open(Pop_file, "w") as file:
            for j, solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j+1, solution.encodes))


        best_index = np.argmin(self.Pop_fitness[:, 0])
        solution = self.Population[best_index]

        # plot the network structure
        Plot_network(solution.dag[0], '{}/{}_conv_dag.png'.format(whole_path, best_index))
        Plot_network(solution.dag[1], '{}/{}_reduc_dag.png'.format(whole_path, best_index))

        # plot the fitness
        fitness_path = whole_path + 'fitness.png'

        line_ = ['o', 'v', '*', '+', '*']
        col_ = ['g', 'r', 'b', 'c', 'm']
        line_type = []
        for i in line_:
            for j in col_:
                line_type.append(j + i + '--')

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), dpi=100)


        x, y = NDSort(self.Pop_fitness, self.popsize)
        # print(x, y)
        for i in range(y):
            # print(np.array(x[0] == i+1))

            l = self.Pop_fitness[np.array(x[0] == i + 1), :].tolist()
            l.sort(key=lambda x: (-x[0], x[1]))
            l = np.array(l)
            ax[0].plot(l[:, 1], l[:, 0], line_type[i], linewidth=1)
        ax[0].set_title('The Pareto Front of Error and Params')
        ax[0].set_ylabel('Error')
        ax[0].set_xlabel('Parameters: MB')


        x, y = NDSort(self.EF_fitness, self.popsize)
        # print(x, y)
        for i in range(y):
            # print(np.array(x[0] == i+1))

            l = self.EF_fitness[np.array(x[0] == i + 1), :].tolist()
            l.sort(key=lambda x: (-x[0], x[1]))
            l = np.array(l)
            ax[1].plot(l[:, 1], l[:, 0], line_type[i], linewidth=1)
        ax[1].set_title('The Pareto Front of Error and FLOPs')
        ax[1].set_xlabel('FLOPs')
        ax[1].set_ylabel('Error')

        fig.savefig(fitness_path)

    def evaluation(self, Pop):
        """
        :param Pop: population
        :return:
        """
        fitness = np.zeros((len(Pop), 4))
        for i, solution in enumerate(Pop):
            logging.info('solution: {0:>2d}'.format(i+1))
            print('solution: {0:>2d}'.format(i+1))
            solution.evaluate(self.args, self.train_queue, self.valid_queue)
            fitness[i] = solution.fitness

        return fitness # error、size_parameters

    def Binary_Envirmental_tour_selection(self):
        """
        binary tournament selection
        :return:
        """
        self.MatingPool, self.tour_index = Mating(self.Population.copy(), self.Pop_fitness, self.FrontValue, self.CrowdDistance)

    def genetic_operation(self):
        """
        genetic operators
        :return:
        """
        offspring_encodes = Generator(self.MatingPool, self.coding, self.popsize)
        print(offspring_encodes)
        offspring_encodes = self.deduplication(offspring_encodes)
        self.offspring = [Individual(i) for i in offspring_encodes]
        self.off_fitness = self.evaluation(self.offspring)

    def first_selection(self):
        """
        :return:
        """
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)


        Population_temp = []
        for i, solution in enumerate(Population):
            if solution.fitness[0] < self.fitness_best + self.threshold:
                Population_temp.append(solution)


        FunctionValue = np.zeros((len(Population_temp),4))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = solution.fitness[:4]


        return Population_temp, FunctionValue

    def Envirment_Selection(self):
        """
        :return:
        """
        Population, FunctionValue = self.first_selection()
        Population, FunctionValue_, FrontValue, CrowdDistance, select_index = EnvironmentSelect(Population, FunctionValue[:, :2], self.popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue_
        self.all_fitness = np.array([FunctionValue[i] for i in select_index])
        self.EF_fitness = self.all_fitness[:, [0, 2]] # error and FLOPs
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance

        self.fitness_best = np.min(self.Pop_fitness[:, 0])

    def deduplication(self, offspring_encodes):
        """
        :param offspring_encodes:
        :return:
        """
        pop_encodes = [i.encodes for i in self.Population]
        dedup_offspring_encodes = []
        for i in offspring_encodes:
            if i not in dedup_offspring_encodes and i not in pop_encodes:
                dedup_offspring_encodes.append(i)
        return dedup_offspring_encodes

    def print_logs(self, since_time=None, initial=False):
        """
        :param since_time:
        :param initial:
        """
        if initial:
            logging.info("*"*40+"Initializing"+"*"*40)
            print("*"*40+"Initializing"+"*"*40)
        else:
            used_time = (time.time()-since_time)/60

            logging.info('*'*40 + '{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min'.format(self.Gen+1, self.Max_Gen, used_time) + '*'*40)

            print('*'*20 + '{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min'.format(self.Gen+1, self.Max_Gen, used_time) + '*'*20)

    def plot_fitness(self):
        """
        :return:
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), dpi=100)
        ax[0].scatter(self.all_fitness[:, 1], self.all_fitness[:, 0])
        ax[0].set_title('Error and Params')
        ax[0].set_xlabel('Parameters: MB')
        ax[0].set_ylabel('Error')
        ax[1].scatter(self.all_fitness[:, 2], self.all_fitness[:, 0])
        ax[1].set_title('Error and FLOPs')
        ax[1].set_xlabel('FLOPs')
        ax[1].set_ylabel('Error')

    def Main_Loop(self):
        since_time = time.time()

        self.print_logs(initial=True) # print log
        self.initialization()
        # self.plot_fitness()


        self.FrontValue = NDSort(self.Pop_fitness, self.popsize)[0]

        self.CrowdDistance = Distance(self.Pop_fitness, self.FrontValue)

        while self.Gen < self.Max_Gen:
            self.print_logs(since_time=since_time)

            self.Binary_Envirmental_tour_selection()
            self.genetic_operation()
            self.Envirment_Selection()

            # self.plot_fitness()
            self.save()
            self.Gen += 1

        # plt.ioff()
        # plt.savefig("{}/final.png".format(self.save_dir))


if __name__=="__main__":

    # ========================================  args  ========================================
    # ********************  common setting  ********************
    parser = argparse.ArgumentParser(description='train search arguments')
    parser.add_argument('--seed', type=int, default=1000) # the seed of architecture initialization
    parser.add_argument('--device', type=str, default='cuda') # GPU device
    parser.add_argument('--save', type=str, default='result') # root dir of results saving

    # ********************  EA setting  ********************
    parser.add_argument('--range_node', type=list, default=[5, 12]) # the nums range of nodes on cell(normal/reduction)
    parser.add_argument('--popsize', type=int, default=20) # the size of population
    parser.add_argument('--Max_Gen', type=int, default=25) # iteration nums of population

    # ********************  dataset setting  ********************
    parser.add_argument('--data', type=str, default="./data/") # root dir of datasets saving
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['spine3, cifar10, cifar100, imagenet']) # the type of dataset
    parser.add_argument('--search_cutout_size', type=int, default=None)  # cutout size 16
    parser.add_argument('--search_autoaugment', action='store_true', default=False) # flag of auto augment
    parser.add_argument('--search_num_work', type=int, default=0, help='the number of the data worker.') # nums of multithreading dataloader

    # ********************  optimization setting  ********************
    parser.add_argument('--search_epochs', type=int, default=25)  # epochs of training on search
    parser.add_argument('--search_lr_max', type=float, default=0.1)  # max lr 0.025 NAO
    parser.add_argument('--search_lr_min', type=float, default=0.001)  # min lr 0 for final training
    parser.add_argument('--search_momentum', type=float, default=0.9) # momentum
    parser.add_argument('--search_l2_reg', type=float, default=1e-4)  # l2 regular term coefficient # 5e-4 for final training
    parser.add_argument('--search_grad_bound', type=float, default=5.0) # grad clip bound
    parser.add_argument('--search_train_batch_size', type=int, default=128) # batch size of training
    parser.add_argument('--search_eval_batch_size', type=int, default=500) # batch size of testing
    parser.add_argument('--search_steps', type=int, default=50000) # steps on global training

    # ********************  structure setting  ********************
    parser.add_argument('--search_use_aux_head', action='store_true', default=False) # flag of aux head
    parser.add_argument('--search_auxiliary_weight', type=float, default=0.4) # aux weight
    parser.add_argument('--search_layers', type=int, default=1) # nums of cell:N, 3 for final Network
    parser.add_argument('--search_keep_prob', type=float, default=0.6) # 0.6 also for final training
    parser.add_argument('--search_drop_path_keep_prob', type=float, default=0.8) # None
    parser.add_argument('--search_channels', type=int, default=16)  # channels of init cell, 24/48 for final training
    parser.add_argument('--search_channels_double', action='store_true', default=False) # False for Cifar, True for ImageNet model

    args = parser.parse_args()
    nums_train = 45000 # cifar10
    # steps=batchs*epochs, train:valid = 9:1(total 50000)
    args.search_steps = int(np.ceil(nums_train / args.search_train_batch_size)) * args.search_epochs
    # dir of searching results save, log file
    args.save = '{}/AEMO_search_{}_{}'.format(args.save, args.dataset, time.strftime("%Y-%m-%d-%H-%M-%S"))

    create_dir(args.save) # create log dir

    # ===================================  logging  ===================================
    # print log info
    log_format = '%(asctime)s %(message)s' # log format
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p') # INFO level

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()])) # key-value

    for name, value in args.__dict__.items():
        print(name, value)

    # ========================================  random seed setting  ========================================
    if not torch.cuda.is_available(): # gpu device
        logging.info('no gpu device available')
        sys.exit(1)

    # random.seed(args.seed)
    # np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    # main
    AEMO_NAS = MOEA(args, visualization=True)
    AEMO_NAS.Main_Loop()
    # MOEA_NAS.initialization()
