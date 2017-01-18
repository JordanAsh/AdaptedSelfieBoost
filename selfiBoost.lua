require 'nn'
require 'optim'
require 'cunn'
require 'torch'
require 'sampler'

-- parse command-line options
opt = lapp[[
--numSamps	      (default 10)	number training samples
--target	      (default 1)      mnist, 2-svhn
--alphaSmooth		      (default 0)	0-nonsmooth 1-smooth
--dSmooth	      (default 0)	0-nonsmooth 1-smooth
--dBoost		      (default 0)	0-boost dist 1- nonboost dist
--seed			(default 0)		random start seed
--rho			(default .125)		stuff
]]

if opt.target == 2 then nc = 3 end

torch.manualSeed(opt.seed)
math.randomseed(opt.seed)

image = require('image')
batchSize = 64

train_file = '/tigress/jordanta/datasets/mnist/train_32x32.t7'
test_file  = '/tigress/jordanta/datasets/mnist/test_32x32.t7'
trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')
trainData['data'] = trainData['data']:cuda()/255
testData['data'] = testData['data']:cuda()/255
dataConfig = {megapatch_w=32, num_dist=25}
svhn = torch.load('/tigress/jordanta/datasets/housenumbers/train_32x32.t7','ascii')

if opt.target == 1 then
	targData = torch.zeros(svhn.X:size(1), 1, 32, 32)
	nc = 1
else
	nc = 3
        targData = torch.zeros(svhn.X:size(1), 3, 32, 32)
end

svhn.X = svhn.X:transpose(3,4)
classes = torch.zeros(10)

for i=1,10 do classes[i] = i end
targLabs = torch.zeros(targData:size(1)):cuda()

for i = 1, targData:size(1) do
    if opt.target == 2 then
        targData[i] = svhn.X[i]
	targLabs[i] = svhn.y[1][i]
    end
end

if opt.target == 1 then
	targData = trainData['data']:cuda()
	targLabs = trainData['labels']:cuda()
end

if opt.target == 3 then
	data = torch.load('/tigress/jordanta/datasets/cifar-10/train.t7')
	targData = data.data:cuda()
	targLabs = data.labels:cuda()
end
targData = targData:cuda()/torch.max(targData)

numSamps = opt.numSamps
classes = {1, 2}
classes = {8, 10}
classes = {7, 9}
classes = {4, 9}
classes = {2, 10}
classes = {5, 8}
classes = {4, 6}
Xtrain = torch.zeros(numSamps, nc, 32, 32):cuda()
Ytrain = torch.zeros(numSamps):cuda()
co = 0
for i = 1, targData:size(1) do 
	for cind, c in pairs(classes) do
		if targLabs[i] == c then
			if co < numSamps then
				Xtrain[co + 1] = targData[i] 
				Ytrain[co + 1] = cind - 1
			end
			co = co + 1
		end
	end
end


Xtest = torch.zeros(co-numSamps + 1, nc, 32, 32):cuda()
Ytest = torch.zeros(co-numSamps + 1):cuda()
co = 1; co2 = 0;
for i = 1, targData:size(1) do
        for cind, c in pairs(classes) do
                if targLabs[i] == c then
                        if co >= numSamps then
                                Xtest[co2+1] = targData[i] 
                                Ytest[co2+1] = (cind-1)
                                co2 = co2 + 1
                        end
                        co = co + 1
                end
        end
end

if opt.target == 2 then targLabs = svhn.y:cuda() end

c = nn.Sequential():cuda()
c:add(nn.SpatialConvolution(nc, 6, 5, 5))
c:add(nn.SpatialMaxPooling(2, 2))
c:add(nn.ReLU())
c:add(nn.SpatialConvolution(6, 16, 5, 5))
c:add(nn.SpatialMaxPooling(2, 2))
c:add(nn.ReLU())
c:add(nn.SpatialConvolution(16, 120, 5, 5))
c:add(nn.ReLU())
c:add(nn.Reshape(120))
c:add(nn.Linear(120, 84))
c:add(nn.ReLU())
c:add(nn.Linear(84, 1))
c:add(nn.Sigmoid())

criterionCE = nn.CrossEntropyCriterion():cuda()
criterionBCE = nn.BCECriterion():cuda()
criterionMSE = nn.MSECriterion():cuda()

params, grads = c:cuda():getParameters()
optimState = {
	learningRate = .001,
	learningRateDecay = 0.01,
	beta1 = .5,
}
 
Y = torch.zeros(batchSize):cuda()
Y2 = Y:clone()
X = torch.zeros(batchSize, nc, 32, 32):cuda()
err = 0
accTrain = 0
accTest = 0
printEvery = 1

D = torch.ones(numSamps)/numSamps
ctest = c:clone()

yNew = torch.zeros(opt.numSamps)
yOld = torch.zeros(opt.numSamps)

for batch = 1, 100 do
	c:training()
	
	-- select random architecture
	r = torch.random()
	
	-- zero gradients
	grads:zero()

        -- train test network
        w = -1
        tcount = 1
	cond1 = false
	cond2 = false
        while not ((w > 0.05 and cond1) and cond2) do
		c:training()
		if opt.dBoost == 1 then
			sInds = sample(D, batchSize)
			for i = 1, batchSize do
				X[i] = Xtrain[sInds[i]]:cuda()
				Y[i] = Ytrain[sInds[i]]
				Y2[i] = yOld[sInds[i]]
			end

			output = c:forward(X)
			err = err + criterionBCE:forward(output, Y)
        		t = criterionBCE:backward(output, Y)
    			c:backward(X, t)

			if batch > 1 then
                                criterionMSE:forward(output, Y2)
                                t = criterionMSE:backward(output, Y2)
                                c:backward(X, t)
                        end
		end

		if opt.dBoost == 0 then
                	for i = 1, batchSize do
				rInd = torch.random(D:size(1))
                        	X[i] = Xtrain[rInd]:cuda()
                        	Y[i] = Ytrain[rInd]
				Y2[i] =  yOld[rInd]
                	end

                	output = c:forward(X)
                	err = err + criterionBCE:forward(output, Y)
    		        t = criterionBCE:backward(output, Y)
	                c:backward(X, t)

		end

		-- get training error
		c:evaluate()
		trainErr = 0
		if opt.alphaSmooth == 0 then
			for i = 1, numSamps do
				yNew[i] = torch.round(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1])
				if Ytrain[i]  ~= torch.round(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1]) then
					trainErr = trainErr + D[i]
				end
			end
		end
		if opt.alphaSmooth == 1 then

                	for i = 1, numSamps do
				yNew[i] = torch.round(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1])
				trainErr = trainErr + (((Ytrain[i] * 2 - 1) * (c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1) + 1) * D[i])/2
			end
			trainErr = 1 - trainErr
		end

		w = .5 * torch.log((1 - trainErr)/(trainErr + 10e-10) + 10e-10)
		tcount = tcount + 1
		optim.adam(function() return 0, grads end, params, optimState)

		Dold = D:clone()		
		Dnew = D:clone()

		tmp = 0
        	for i=1, numSamps do
                	Dnew[i] = Dold[i] * torch.exp(-1 * w * sign(Ytrain[i] * 2 - 1) * sign(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1))
        		--tmp1 = .5 * (sign(Ytrain[i] * 2 - 1)  - sign(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1)) ^ 2
			--tmp2 = -1 * sign(Ytrain[i] * 2 - 1)  * ((yOld[i] * 2 - 1) -  sign(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1))
			--print(tmp1, tmp2)
        		tmp1 = .5 * ((Ytrain[i] * 2 - 1)  - (c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1)) ^ 2
			tmp2 = 1 * (Ytrain[i] * 2 - 1)  * ((yOld[i] * 2 - 1) -  (c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1))
			tmp = tmp + Dnew[i] / Dold[i] * (tmp1 + tmp2)
		end
		tmp = tmp / torch.sum(Dnew)
        	Dnew = Dnew/torch.sum(Dnew)
		
		-- get conditional size
		cond1 = true
		--print(opt.rho, tmp)
		if batch > 1 then cond1 = (tmp < (-1 * opt.rho)) end		
		--cond1 = true
		cond2 = true
		if batch > 2 then
			for i = 1, numSamps do
				if torch.round(yOld[i]) == Y[i] then
					if torch.round(yNew[i]) ~= Y[i] then
						cond2 = false
					end
				end
			end
		end
		
		if opt.dBoost == 0 then break end
	end
	yOld = yNew:clone()	


	-- evaluate on training data
	c:evaluate()
	for i = 1, batchSize do
		rInd = torch.random(numSamps)
                X[i] = Xtrain[rInd]:cuda()
        	Y[i] = Ytrain[rInd]
        end
	output = c:forward(X)
	accTrain = torch.sum((torch.round(output):cuda() - Y):eq(0)) + accTrain

	-- evaluate on testing data
	for i = 1, batchSize do
                rInd = torch.random(Xtest:size(1))
                X[i] = Xtest[rInd]:cuda()
                Y[i] = Ytest[rInd]
        end
	output = c:forward(X)
	accTest = torch.sum((torch.round(output):cuda() - Y):eq(0)) + accTest

	-- report results
	if batch % printEvery == 0 then
		print(batch, trainErr, tcount, optimState.learningRate, err/printEvery/batchSize, accTrain/printEvery/batchSize, accTest/printEvery/batchSize)
		err      = 0
		accTrain = 0
		accTest  = 0
	end

	-- update D
	c:training()
	for i=1, numSamps do
               	torch.manualSeed(r)
               	D[i] = D[i] * torch.exp(-1 * w * sign(Ytrain[i] * 2 - 1) * sign(c:forward(Xtrain[i]:resize(1, nc, 32, 32))[1][1] * 2 - 1))
        end
	D = D/torch.sum(D)
	c:training()
end

