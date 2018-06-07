import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.model_path = opt.model_path

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.model_path, save_filename)
        print(save_path)

        network.load_state_dict(torch.load(save_path))

        # state_dict = torch.load(save_path)
        # # from collections import OrderedDict
        # # new_state_dict = OrderedDict()
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     # print(k)
        #     if k.find('conv') != -1:
        #         name = k[6:]
        #     else:
        #         name = k
        #     # print(name + '\n')
        #     new_state_dict[name] = v
        # # load params
        # for k, _ in network.state_dict().items():
        #     print(k)
        # network.load_state_dict(new_state_dict)
