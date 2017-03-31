local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolutionMM
local Activation = nn.ReLU
local Pooling = nn.SpatialMaxPooling
local Linear = nn.Linear
local Conorm = nn.SpatialContrastiveNormalization

local model = nn.Sequential()

--First Conv Layer
model:add(Convolution(3, 108, 5, 5, 1, 1, 2, 2))
model:add(Activation())
model:add(Pooling(2, 2, 2, 2))
model:add(Conorm(108, image.gaussian(5)))

--Second Conv Layer
model:add(Convolution(108, 200, 5, 5, 1, 1, 2, 2))
model:add(Activation())
model:add(Pooling(2, 2, 2, 2))
model:add(Conorm(200, image.gaussian(5)))

--FC Layer
model:add(nn.View(12800))
model:add(Linear(12800, 100))
model:add(Activation())

--Output Layer
model:add(nn.View(100))
model:add(Linear(100, 43))

return model
