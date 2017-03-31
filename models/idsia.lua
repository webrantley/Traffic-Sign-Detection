local nn = require 'nn'
local image = require 'image'


local Convolution = nn.SpatialConvolution
local Activation = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Conorm = nn.SpatialContrastiveNormalization

local model  = nn.Sequential()

--First Conv layer
model:add(Convolution(3, 100, 7, 7))
model:add(Activation())
model:add(Max(2, 2, 2, 2))
model:add(Conorm(100, image.gaussian1D(5)))

--Second Conv layer
model:add(Convolution(100, 150, 4, 4))
model:add(Activation())
model:add(Max(2, 2, 2, 2))
model:add(Conorm(150, image.gaussian1D(5)))

--Third Conv layer
model:add(Convolution(150, 250, 4, 4))
model:add(Activation())
model:add(Max(2, 2, 2, 2))
model:add(Conorm(250, image.gaussian1D(5)))

--First Fully Connected layer
--model:add(nn.Reshape(250*3*3))
--model:add(Linear(250*3*3, 300))
model:add(View(250))
model:add(Linear(250, 300))
model:add(Activation())

--Output layer
model:add(Linear(300, 43))

return model
