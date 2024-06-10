from matplotlib import pyplot as plt
from argparse import ArgumentParser
from matplotlib.axes import Axes
from torchvision import utils
import numpy as np
import torch
import time
import os

def Activation(activation: str):
    activation = activation.lower()
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'silu':
        return torch.nn.SiLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    else:
        return torch.nn.ReLU()
    

class Losses(object):
    def __init__(self, losses_name: list):
        '''
        Init name of all loss, and create 2 lists for loss:
            1. self_dict[name] for loss of each iteration.
            2. self_dict[f'{name}_epoch'] for loss of each epoch.
        '''
        self.losses_name = losses_name
        self._total_loss = 0
        self.epoch_loss: dict[str, list] = {}
        self.steps = 0
        for name in losses_name:
            setattr(self, name, [])
            self.epoch_loss[f'{name}_epoch'] = []

    def avg_loss(self):
        '''
        Calculate the avg loss of each epoch, and return the loss string.
        add the avg loss to the list {name}_avg.
        ex: {'loss_avg': 0.123, 'ce_loss_avg': 0.456, ...}
        return: ' loss:0.123 | ce_loss:0.456 | ...'
        '''
        output_str = ''
        for i, name in enumerate(self.losses_name):
            setattr(self, f'{name}_avg', np.mean(self.__dict__[name]))
            if i == 0:
                self._total_loss = self.__dict__[f'{name}_avg']
            output_str += f' {name}:{self.__dict__[f"{name}_avg"]:.3f} |'
        return output_str

    def save_loss(self):
        '''
        Save the avg loss of every epoch to the list {name}_epoch, and clear the list of {name}.
        ex: {'loss_epoch': [0.123, 0.456, ...], 'ce_loss_epoch': [0.789, 0.012, ...], ...}
        '''
        self_dict = self.__dict__
        for i, name in enumerate(self.losses_name):
            self.epoch_loss[f'{name}_epoch'].append(np.mean(self_dict[name]))
            if i == 0:
                self._total_loss = self.epoch_loss[f'{name}_epoch'][-1]
            self_dict[name] = []
        return self.epoch_loss

    def write_log_file(self):
        '''
        Write the avg loss of every epoch to the log dict.
        return: {'loss': 0.123, 'ce_loss': 0.456, ...}
        '''
        log_dict = {}
        for element in self.__dict__:
            if '_avg' in element:
                log_dict[element.replace('_avg', '')] = self.__dict__[element]
        return log_dict

    def push_loss(self, loss_list: list):
        '''
        Push the loss of each iteration to the list {name}.
        '''
        self.steps += 1
        for i, name in enumerate(self.losses_name):
            self.__dict__[name].append(loss_list[i])

class Training_Log():
    def __init__(self, model_name: str = 'My_Model',
                 save_loss_fig: bool = True, 
                 save_weight: bool = True, weight_mode: str = 'min', save_criteria: str = 'val', weight_start: int = 30, 
                 step_mode: str = 'epoch'):
        '''
        Args:
            model_name (str): The name of the model.
            save_loss_fig (bool): Save the loss figure or not.
            save_weight (bool): Save the weight of the model or not.
            weight_mode (str): The mode of saving the weight. 'min' for saving the weight with the minimum loss, 'max' for saving the weight with the maximum loss.
            save_criteria (str): The criteria of saving the weight. 
                                 'val' for saving the weight with the 'weight_mode' of validation loss, 
                                 'test' for saving the weight with the 'weight_mode' of test loss,
                                 'metric' for saving the weight with the 'weight_mode' of metric,
                                 'train' for saving the weight with the 'weight_mode' of training loss.
            weight_start (int): The epoch to start saving the weight.
            step_mode (str): The mode of the step. 'epoch' for step in each 'epoch', 'step' for the step in 'iteration'.
        '''
        self.arg_parse(model_name)
        self.save_init()
        self.save_loss_fig = save_loss_fig
        self.save_criteria = save_criteria
        self.save_weight = save_weight
        self.weight_start = weight_start
        self._all_loss = {}
        self.step_mode = step_mode
        self.epochs = 0
        self.steps_stone = []
        self.warning = False
        
        if weight_mode not in ['min', 'max']:
            raise ValueError(f'\033[0;31m\033[1mInvalid weight mode: \"{weight_mode}\", support mode: [\'min\', \'max\']\033[0m')
        if weight_mode == 'min':
            self.best = float('inf')
            self.weight_mode = lambda input, best: input < best
        elif weight_mode == 'max':
            self.best = float('-inf')
            self.weight_mode = lambda input, best: input > best

    def save_init(self,):
        # Init the save path of the loss figure and the weight.
        if not os.path.exists(self.project_name):
            os.makedirs(self.project_name)
        WEIGHT_SAVE = f"{self.project_name}/ck"
        if not os.path.exists(WEIGHT_SAVE):
            os.makedirs(WEIGHT_SAVE)
        STEP_SAVE = f"{self.project_name}/out"
        if not os.path.exists(STEP_SAVE):
            os.makedirs(STEP_SAVE)
        self.STEP_SAVE = STEP_SAVE
        self.WEIGHT_SAVE = WEIGHT_SAVE

    def init_log_file(self, loss_type: str):
        # Init the log file of each loss and write the training setting.
        with open(f'./{self.project_name}/{loss_type}_log.log', '+a') as loss_log:
            loss_log.write(f'Training setting: \n\t{{learning rate:{self.lr}, batch size:{self.batch}, optimizer:\'{self.optimizer}\', total epochs:{self.total_epochs}}}\n')
            loss_log.write(f'\t{{autocast:\'{self.auto_cast}\', compile:\'{self.compile}\', resume:\'{self.resume}\'}}\n')
            loss_log.write(f'\t{{gradient accumulation: {self.grad_accum}}}\n')

    def init_loss(self, **kwargs):
        # Init the loss object of each loss.
        '''
        Args:
            train_losses (list): The name of the training loss.
            test_losses (list): The name of the test loss.
            val_losses (list): The name of the validation loss.
            metrics (list): The name of the metric.
            ex: train_losses = ['total_loss', 'ce_loss', 'dice_loss']
                log.init_loss(train_losses = train_losses)

        Note:
            The 'total loss' or the score 'used to determine whether the model is stored' 
            should be the 'first element' in the list.
        '''
        support_type = ['train_losses', 'test_losses', 'val_losses', 'metrics']
        for loss_type, loss_name in kwargs.items():
            if loss_type not in support_type:
                raise ValueError(f'\033[0;31m\033[1mInvalid loss type: \"{loss_type}\", support type: {support_type}\033[0m')
            self._all_loss[loss_type] = {'names': loss_name, 'loss_boj': Losses(loss_name)}
            self.init_log_file(loss_type)
        
        # Bind the loss object to the class. Just for the convenience of the user.
        if 'train_losses' in self._all_loss:
            self.train_loss: Losses = self._all_loss['train_losses']['loss_boj']
        if 'test_losses' in self._all_loss:
            self.test_loss: Losses = self._all_loss['test_losses']['loss_boj']
        if 'val_losses' in self._all_loss:
            self.val_loss: Losses = self._all_loss['val_losses']['loss_boj']
        if 'metrics' in self._all_loss:
            self.metrics: Losses = self._all_loss['metrics']['loss_boj']

    def arg_parse(self, model_name: str = 'My_Model'):
        # Parse the argument.
        # You can change the default value and add your own argument here.
        parser = ArgumentParser()
        random_name = time.strftime(f"%Y%m%d_%H%M%S", time.localtime()) 
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('-B', '--batch', type=int, default=64, help='batch size')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('-E', '--epochs', type=int, default=600, help='epochs')
        parser.add_argument('-NK', "--num-workers", type=int, default=12, help='num_workers')
        parser.add_argument('-PN', '--project', type=str, default=random_name, help='project name')
        parser.add_argument('-C', '--compile', help='compile model', action='store_true')
        parser.add_argument('-A', '--auto-cast', help='using auto-cast', action='store_true')
        parser.add_argument('-O', '--optimizer', type=str, default='SGD', help='optimizer')
        parser.add_argument('-R', '--resume', type=str, help="reload model from given path", default=None)
        parser.add_argument('-GA', '--grad-accum', type=int, default=1, help='gradient accumulation steps, set to 1 to disable')
        args = parser.parse_args()
        self.project_name = model_name + '_' + args.project

        # No need to do the following. Just for the convenience of the user.
        self.gpu = args.gpu
        self.batch = args.batch
        self.lr = args.lr
        self.total_epochs = args.epochs
        self.num_workers = args.num_workers
        self.compile = args.compile
        self.auto_cast = args.auto_cast
        self.optimizer = args.optimizer
        self.resume = args.resume
        self.grad_accum = args.grad_accum

    def loss_txt_log(self):
        # Write the loss to the log file.
        self.steps_stone.append(self.train_loss.steps)
        for loss_type, loss_dict in self._all_loss.items():
            log_content = loss_dict['loss_boj'].write_log_file()
            with open(f'./{self.project_name}/{loss_type}_log.log', '+a') as loss_log:
                log_str = '{'
                for k, v in log_content.items():
                    log_str += f'\'{k}\': {v:.10f},  '
                log_str += '}'
                step_line = f' steps {self.train_loss.steps:6d}' if self.step_mode == 'step' else ''
                log_line = f'epoch {self.epochs:4d}' + step_line + ':' + log_str + '\n'
                loss_log.write(log_line)

    def loss_fig_log(self):
        # Save the loss figure.
        # Init the figure and set the size, title, and background color.
        fig_height = 0.2 + 6 * len(self._all_loss)
        fig = plt.figure(figsize=(16, fig_height))
        self.lightGray = (0.9, 0.9, 0.9)
        fig.suptitle(f'Loss of Epoch: {self.epochs}',color=self.lightGray, fontsize=18, y=0.92)
        fig.patch.set_facecolor((0.12, 0.12, 0.12))

        # Add the subplot for each type of loss.
        plot_num = len(self._all_loss)*100 + 11 # 211, 311, 411, ...
        for loss_type, loss_dict in self._all_loss.items():
            ax = fig.add_subplot(plot_num)
            ax = self.ax_setting(ax, title=loss_type, x_label=self.step_mode, y_label='value')
            all_loss = loss_dict['loss_boj'].save_loss()
            for one_loss in loss_dict['names']:
                x = np.arange(len(all_loss[one_loss + '_epoch'])) if self.step_mode == 'epoch' else self.steps_stone
                ax.plot(
                    x, 
                    all_loss[one_loss + '_epoch'], 
                    label=one_loss)
            plt.grid(color=(0.25, 0.25, 0.28), linestyle="-", linewidth=1)
            plt.legend()
            plot_num += 1

        plt.savefig(f'{self.project_name}/loss_fig.png')
        plt.close()

    def ax_setting(self, ax: Axes, title: str = 'loss', x_label: str = 'epoch', y_label='loss'):
        # Set the color of the ax.
        ax.set_title(title, color=self.lightGray, fontsize=16)
        ax.set_facecolor((0.2, 0.2, 0.2))
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.xaxis.label.set_color(self.lightGray)
        ax.yaxis.label.set_color(self.lightGray)
        ax.tick_params(axis="x", colors=self.lightGray)
        ax.tick_params(axis="y", colors=self.lightGray)
        ax.spines[:].set_color(self.lightGray)
        return ax
    
    def get_criteria(self):
        # Get the criteria of saving the weight.
        if self.save_criteria == 'val':
            return self.val_loss._total_loss
        elif self.save_criteria == 'test':
            return self.test_loss._total_loss
        elif self.save_criteria == 'metric':
            return self.metrics._total_loss
        else:
            return self.val_loss._total_loss
        
    def step(self, epochs=None, result_img=None, net_weight=None, optimizer_state=None):
        """
        Args:
            epochs (int): The current epoch.
            result_img (Tensor): Image of the result of the model. shape: [batch, channel, height, width]
            net_weight (OrderedDict): The state_dict of the model.
            optimizer_state (OrderedDict): The state_dict of the optimizer.
        Note:
            1. If the step_mode is 'epoch', the epochs will be accumulated automatically. 
               If the step_mode is 'step', the epochs should be input manually for correct epoch.
            2. The result_img will be saved to the ./<modelname>/out/ folder.
            3. The net_weight and optimizer_state will be saved to the ./<modelname>/ck/ folder.
        """
        if self.step_mode == 'step' and epochs is not None:
            self.epochs = epochs
        else:
            self.epochs += 1

        self.loss_txt_log() # Write the loss to the log file.

        if self.save_loss_fig:
            # Save the loss figure.
            self.loss_fig_log()

        if result_img is not None:
            # Save the result image. More function will be added in the future.
            utils.save_image(result_img.float(), f"{self.STEP_SAVE}/{self.epochs:04d}.png", pad_value=0.3)
            
        if self.save_weight and net_weight is not None:
            compiled = '_compiled' if self.compile else ''

            criteria = self.get_criteria()
            if self.weight_mode(criteria, self.best) and self.epochs >= self.weight_start:
                # Save the weight of the best model when epoch >= weight_start.
                torch.save(net_weight, f"{self.WEIGHT_SAVE}/ck_{str(self.epochs).zfill(4)}{compiled}.pt")
                
                if optimizer_state is not None:
                    # Save the optimizer state_dict of the best model.
                    torch.save(optimizer_state, f"{self.WEIGHT_SAVE}/ck_{str(self.epochs).zfill(4)}{compiled}_optim.pt")

                self.best = self.test_loss._total_loss

            # Save the weight of the last model.
            torch.save(net_weight, f"{self.WEIGHT_SAVE}/ck_last{compiled}.pt")
            torch.save(optimizer_state, f"{self.WEIGHT_SAVE}/ck_last{compiled}_optim.pt")

        elif self.save_weight and not self.warning:
            if net_weight == None :
                print("\033[0;33m[WARNING] No weight saved. Please check your input model weight!\033[0m")
            if optimizer_state == None:
                print("\033[0;33m[WARNING] No optimizer state_dict saved. Please check your input optimizer!\033[0m")
            self.warning = True
