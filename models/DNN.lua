local nn = require 'nn'
local image = require 'image'
--local cunn = require 'cunn'
--local cudnn = require 'cudnn'
local optim = require 'optim'


print("making a deeper net")
local Convolution = nn.SpatialConvolutionMM
local Activation = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local BatchNorm = nn.SpatialBatchNormalization

local model  = nn.Sequential()

--First Conv layer
model:add(Convolution(3, 150, 7, 7))
model:add(BatchNorm(150))
model:add(Activation())
model:add(Max(2, 2, 2, 2))

--Second Conv layer
model:add(Convolution(150, 200, 4, 4))
model:add(BatchNorm(200))
model:add(Activation())
model:add(Max(2, 2, 2, 2))

--Third Conv layer
model:add(Convolution(200, 300, 4, 4))
model:add(BatchNorm(300))
model:add(Activation())
model:add(Max(2, 2, 2, 2))

--First Fully Connected layer
--model:add(nn.Reshape(250*3*3))
--model:add(Linear(250*3*3, 300))
model:add(View(2700))
model:add(Linear(2700, 350))
model:add(nn.BatchNormalization(350))
model:add(Activation())

--Output layer
model:add(Linear(350, 43))

return model
