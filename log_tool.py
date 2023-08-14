import torch
from matplotlib import pyplot as plt
from torchvision import utils
import numpy as np
import os

class Losses(object):
    def __init__(self, losses_name:list):
        '''
        Init name of all loss, and create a lists for loss:
            self_dict[name] for loss of each iteration.
        '''
        self.losses_name = losses_name
        self._total_loss = 0
        self_dict = self.__dict__
        self.all_loss = {}
        for name in losses_name:
            self_dict[name] = []
            self.all_loss[f'{name}_all'] = []
        
    def avg_loss(self):
        self_dict = self.__dict__
        output_str = ''
        for i, name in enumerate(self.losses_name):
            self_dict[f'{name}_avg'] = np.mean(self_dict[name])
            if i == 0:
                self._total_loss = self_dict[f'{name}_avg']
            output_str += f' {name}:{self_dict[f"{name}_avg"]:.3f} |'
        return output_str

    def save_loss(self):
        '''
        Save the loss of every epoch to the list {name}_all, and clear the list of {name}.
        '''
        self_dict = self.__dict__
        for i, name in enumerate(self.losses_name):
            self.all_loss[f'{name}_all'].append(np.mean(self_dict[name]))
            if i == 0:
                self._total_loss = self.all_loss[f'{name}_all'][-1]
            self_dict[name] = []
        return self.all_loss

    def write_log_file(self):
        '''
        Return a dict catain all loss to write log file.
        '''
        self_dict = self.__dict__
        log_dict = {}
        for element in self_dict:
            if '_avg' in element:
                log_dict[element.replace('_avg', '')] = self_dict[element]
        return log_dict
    
    def push_loss(self, loss_list:list):
        '''
        Push the loss of each iteration to the list {name}.
        '''
        self_dict = self.__dict__
        for i, name in enumerate(self.losses_name):
            self_dict[name].append(loss_list[i])
    
def save_init(PROJECT_NAME = "Your_Project_Name"):
    if not os.path.exists(PROJECT_NAME):
        os.makedirs(PROJECT_NAME)
    WEIGHT_SAVE = f"{PROJECT_NAME}/check_point"
    if not os.path.exists(WEIGHT_SAVE):
        os.makedirs(WEIGHT_SAVE)
    STEP_SAVE = f"{PROJECT_NAME}/output_fig"
    if not os.path.exists(STEP_SAVE):
        os.makedirs(STEP_SAVE)
    LOG_PATH = f"{PROJECT_NAME}/loss_log.log"
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    return WEIGHT_SAVE, STEP_SAVE

class Training_Log():
    def __init__(self, PROJECT_NAME:str, train_lossesname:list, test_lossesname:list,
                 save_fig:bool=True, save_weight:bool=True, weight_start:int=30):
        self.project_name = PROJECT_NAME
        self.train_loss = Losses(train_lossesname)
        self.test_loss = Losses(test_lossesname)
        self.train_loss_name = train_lossesname
        self.test_loss_name = test_lossesname
        self.WEIGHT_SAVE, self.STEP_SAVE = save_init(PROJECT_NAME)
        self.save_fig = save_fig
        self.save_weight = save_weight
        self.weight_start = weight_start
        self.best = 10000
        self.epochs = 1
    
    def loss_txt_log(self, eps):
        log_content = self.train_loss.write_log_file()
        with open(f'./{self.project_name}/train_loss_log.log', '+a') as loss_log:
            loss_log.write(f'epoch {eps:4d}:' + str(log_content).replace(',',',\t') + '\n')

        log_content = self.test_loss.write_log_file()
        with open(f'./{self.project_name}/test_loss_log.log', '+a') as loss_log:
            loss_log.write(f'epoch {eps:4d}:' + str(log_content).replace(',',',\t') + '\n')
    
    def loss_fig_log(self, eps, all_train_loss, all_test_loss):
        fig = plt.figure(figsize=(16, 10.5))
        lightGray = (0.9, 0.9, 0.9)
        fig.suptitle(f'Loss of Epoch: {eps}',color =lightGray, fontsize=18)
        fig.patch.set_facecolor((0.12, 0.12, 0.12))
        ax = fig.add_subplot(211)
        ax.set_title('train_loss', color =lightGray, fontsize = 16)
        ax.set_facecolor((0.2, 0.2, 0.2))
        ax.set_xlabel("epoch",fontsize = 13)
        ax.set_ylabel("loss",fontsize = 13)
        ax.xaxis.label.set_color(lightGray)
        ax.yaxis.label.set_color(lightGray)
        ax.tick_params(axis="x",colors=lightGray)
        ax.tick_params(axis="y",colors=lightGray)
        ax.spines[:].set_color(lightGray)
        for one_loss in self.train_loss_name:
            one_loss += '_all'
            ax.plot(np.arange(len(all_train_loss[one_loss])), all_train_loss[one_loss], label=one_loss[:-4])
        plt.grid(color = (0.25,0.25,0.28), linestyle="-", linewidth = 1)
        plt.legend()

        ax = fig.add_subplot(212)
        ax.set_title('test_loss', color =lightGray, fontsize = 16)
        ax.set_facecolor((0.2,0.2,0.2))
        ax.set_xlabel("epoch",fontsize = 13)
        ax.set_ylabel("loss",fontsize = 13)
        ax.xaxis.label.set_color(lightGray)
        ax.yaxis.label.set_color(lightGray)
        ax.tick_params(axis="x",colors=lightGray)
        ax.tick_params(axis="y",colors=lightGray)
        ax.spines[:].set_color(lightGray)
        for one_loss in self.test_loss_name:
            one_loss += '_all'
            ax.plot(np.arange(len(all_test_loss[one_loss])), all_test_loss[one_loss], label=one_loss[:-4])
        plt.grid(color = (0.25,0.25,0.28), linestyle="-", linewidth = 1)
        plt.legend()

        plt.savefig(f'{self.project_name}/loss_fig.png')
        plt.close()
    
    def step(self, result_img = None, net_weight = None):
        all_train_loss = self.train_loss.save_loss()
        all_test_loss = self.test_loss.save_loss()
        self.loss_txt_log(self.epochs)
        if self.save_fig:
            self.loss_fig_log(self.epochs, all_train_loss, all_test_loss)
        if result_img is not None:
            utils.save_image(result_img.float(), f"{self.STEP_SAVE}/{self.epochs:04d}.png")
        if self.save_weight and net_weight is not None:
            if self.test_loss._total_loss < self.best and self.epochs >= self.weight_start :
                torch.save(net_weight, f"{self.WEIGHT_SAVE}/ck_{str(self.epochs).zfill(4)}.pt")
                self.best = self.test_loss._total_loss
        self.epochs += 1
