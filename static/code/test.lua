require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'torch'


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i --alphabet的第i个字符对应i e.g. dict['a']=1
end

cmd = torch.CmdLine()
cmd:option('-txt_limit',0,'if 0 then use all available text. Otherwise limit the number of documents per class')
cmd:option('-gpuid',0,'gpu to use')
cmd:option('-savefile','./embeddings','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-model','./models/embeddings.t7','model to load. If blank then above options will be used.')
cmd:option('-snt','this is a bird','a sentence about a bird')
opt = cmd:parse(arg)

model = torch.load(opt.model)

if opt.gpuid >= 0 then --选定GPU id， lua的下标一般从1开始
    cutorch.setDevice(opt.gpuid+1)
end

local doc_length = model.opt.doc_length --文件长度
local protos = model.protos --原型
protos.enc_doc:evaluate()
protos.enc_image:evaluate()
print(opt.snt)

function extract_txt_char()
    snt = opt.snt
    txt = torch.zeros(1,1,201)
    for i = 1, #snt do
        txt[1][1][i] = dict[snt:sub(i,i)]
    end
    txt = txt:reshape(txt:size(1)*txt:size(2),txt:size(3)):float():cuda()
    if opt.txt_limit > 0 then
        local actual_limit = math.min(txt:size(1), opt.txt_limit)
        txt_order = torch.randperm(txt:size(1)):sub(1,actual_limit)
        local tmp = txt:clone()
        for i = 1,actual_limit do
            txt[{i,{}}]:copy(tmp[{txt_order[i],{}}])
        end
        txt = txt:narrow(1,1,actual_limit)
    end
    local txt_mat = torch.zeros(txt:size(1), txt:size(2), #alphabet)
    for i = 1,txt:size(1) do
        for j = 1,txt:size(2) do
            local on_ix = txt[{i, j}]
            if on_ix == 0 then
                break
            end
            txt_mat[{i, j, on_ix}] = 1
        end
    end
    txt_mat = txt_mat:float():cuda()
    local out = protos.enc_doc:forward(txt_mat)
    return out:float()
end

embedding = extract_txt_char()
local savefile = string.format("%s/txt_embedding.t7", opt.savefile)
torch.save(savefile,embedding)
print(embedding:size())




