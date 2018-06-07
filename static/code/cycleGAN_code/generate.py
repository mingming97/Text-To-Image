import torch.utils.data
from data.dataset import SingleDataset
from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from models.model import TestModel
from options.base_options import mkdir


opt = TestOptions().parse()
dataset = CreateDataLoader(opt)
dataset = dataset.load_data()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = TestModel()
model.initialize(opt)

new_path = opt.dataroot + 'gen_pic/'
mkdir(new_path)

for i, data in enumerate(dataset, 0):
	model.set_input(data)
	fake = model.gen()
	new_img_path = opt.dataroot + 'gen_pic/' + str(i) + '.png'
	fake.save(new_img_path)
	print('ok' + str(i))
