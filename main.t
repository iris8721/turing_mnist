setscreen ("graphics:max;max,nobuttonbar,nocursor,offscreenonly")

var fontID : int
fontID := Font.New ("serif:12")
assert fontID > 0

const INPUT_NODES : int := 784
const HIDDEN_NODES : int := 128
const OUTPUT_NODES : int := 10
const INITIAL_LEARNING_RATE : real := 0.1
var LEARNING_RATE : real := INITIAL_LEARNING_RATE
const EPOCHS : int := 10
const MODEL_FILE : string := "mnist_model.dat"

var trainingSet : int
var pixel : int
var currentByte : string (255)
var currentChar : string (1)
var colourCode : int
var imageCount : int
var currentImage : int
var mnistArray : array 0 .. 59999, 0 .. 784 of real

var inputLayer : array 1 .. INPUT_NODES of real
var hiddenLayer : array 1 .. HIDDEN_NODES of real
var outputLayer : array 1 .. OUTPUT_NODES of real
var weightsIH : array 1 .. INPUT_NODES, 1 .. HIDDEN_NODES of real
var weightsHO : array 1 .. HIDDEN_NODES, 1 .. OUTPUT_NODES of real
var biasH : array 1 .. HIDDEN_NODES of real
var biasO : array 1 .. OUTPUT_NODES of real

procedure initializeWeights
    for i : 1 .. INPUT_NODES
	for j : 1 .. HIDDEN_NODES
	    weightsIH(i, j) := Rand.Real - 0.5
	end for
    end for
    for i : 1 .. HIDDEN_NODES
	for j : 1 .. OUTPUT_NODES
	    weightsHO(i, j) := Rand.Real - 0.5
	end for
	biasH(i) := Rand.Real - 0.5
    end for
    for i : 1 .. OUTPUT_NODES
	biasO(i) := Rand.Real - 0.5
    end for
end initializeWeights

function sigmoid (x : real) : real
    result 1 / (1 + exp(-x))
end sigmoid

function sigmoidDerivative (x : real) : real
    result x * (1 - x)
end sigmoidDerivative

procedure loadData
    var fontID : int := Font.New ("serif:12")
    assert fontID > 0

    colourCode := RGB.AddColour (0, 0, 0)
    imageCount := 0

    open : trainingSet, "mnist_train.csv", get

    var imageLimit : int := 1000

    loop
	cls
	put "processing images... " + intstr(imageCount)
	View.Update
	exit when eof (trainingSet) or imageCount = imageLimit
	get : trainingSet, currentChar : 1 % label
	mnistArray(imageCount, 0) := strint (currentChar)
	get : trainingSet, currentChar : 1 % first comma
	for i : 1 .. 784
	    currentByte := ""
	    loop
		get : trainingSet, currentChar : 1
		exit when currentChar = "," or currentChar = "\n"
		currentByte := currentByte + currentChar
	    end loop
	    mnistArray (imageCount, i) := strint (currentByte) / 255.0
	end for
	imageCount += 1
    end loop
    cls
    View.Update
end loadData

procedure decayLearningRate (epoch : int)
    LEARNING_RATE := INITIAL_LEARNING_RATE / (1 + 0.01 * epoch)
end decayLearningRate

procedure saveModel
    var modelFile : int
    open : modelFile, MODEL_FILE, put

    for i : 1 .. INPUT_NODES
	for j : 1 .. HIDDEN_NODES
	    put : modelFile, weightsIH(i, j)
	end for
    end for

    for i : 1 .. HIDDEN_NODES
	for j : 1 .. OUTPUT_NODES
	    put : modelFile, weightsHO(i, j)
	end for
	put : modelFile, biasH(i)
    end for

    for i : 1 .. OUTPUT_NODES
	put : modelFile, biasO(i)
    end for

    close : modelFile
    put "model saved to ", MODEL_FILE
end saveModel

procedure matrixMultiply (a : array 1 .. * of real, b : array 1 .. *, 1 .. * of real, var res : array 1 .. * of real)
    for i : 1 .. upper(res)
	res(i) := 0
	for j : 1 .. upper(a)
	    res(i) += a(j) * b(j, i)
	end for
    end for
end matrixMultiply

procedure matrixVectorMultiply (a : array 1 .. *, 1 .. * of real, b : array 1 .. * of real, var res : array 1 .. * of real)
    for i : 1 .. upper(res)
	res(i) := 0
	for j : 1 .. upper(b)
	    res(i) += a(j, i) * b(j)
	end for
    end for
end matrixVectorMultiply

procedure forwardPropagation
    var tempHidden : array 1 .. HIDDEN_NODES of real
    matrixMultiply(inputLayer, weightsIH, tempHidden)
    for i : 1 .. HIDDEN_NODES
	hiddenLayer(i) := sigmoid(tempHidden(i) + biasH(i))
    end for

    var tempOutput : array 1 .. OUTPUT_NODES of real
    matrixVectorMultiply(weightsHO, hiddenLayer, tempOutput)
    for i : 1 .. OUTPUT_NODES
	outputLayer(i) := sigmoid(tempOutput(i) + biasO(i))
    end for
end forwardPropagation

procedure backpropagation (target : array 1 .. * of real)
    var outputErrors, outputDeltas : array 1 .. OUTPUT_NODES of real
    var hiddenErrors, hiddenDeltas : array 1 .. HIDDEN_NODES of real

    for i : 1 .. OUTPUT_NODES
	outputErrors(i) := target(i) - outputLayer(i)
	outputDeltas(i) := outputErrors(i) * sigmoidDerivative(outputLayer(i))
    end for

    for i : 1 .. HIDDEN_NODES
	hiddenErrors(i) := 0
	for j : 1 .. OUTPUT_NODES
	    hiddenErrors(i) += weightsHO(i, j) * outputDeltas(j)
	end for
	hiddenDeltas(i) := hiddenErrors(i) * sigmoidDerivative(hiddenLayer(i))
    end for

    for i : 1 .. HIDDEN_NODES
	for j : 1 .. OUTPUT_NODES
	    weightsHO(i, j) += LEARNING_RATE * outputDeltas(j) * hiddenLayer(i)
	end for
    end for

    for i : 1 .. INPUT_NODES
	for j : 1 .. HIDDEN_NODES
	    weightsIH(i, j) += LEARNING_RATE * hiddenDeltas(j) * inputLayer(i)
	end for
    end for

    for i : 1 .. OUTPUT_NODES
	biasO(i) += LEARNING_RATE * outputDeltas(i)
    end for

    for i : 1 .. HIDDEN_NODES
	biasH(i) += LEARNING_RATE * hiddenDeltas(i)
    end for
end backpropagation

procedure train
    var bestLoss : real := 1000000
    var epochsWithoutImprovement : int := 0
    const EARLY_STOPPING_PATIENCE : int := 5
    const BATCH_SIZE : int := 32

    for epoch : 1 .. EPOCHS
	var totalEpochLoss : real := 0
	decayLearningRate(epoch)

	for batchStart : 0 .. imageCount - 1 by BATCH_SIZE
	    var batchEnd : int := min(batchStart + BATCH_SIZE - 1, imageCount - 1)
	    var batchLoss : real := 0

	    for i : batchStart .. batchEnd
		for j : 1 .. INPUT_NODES
		    inputLayer(j) := mnistArray(i, j)
		end for
		var target : array 1 .. OUTPUT_NODES of real
		for j : 1 .. OUTPUT_NODES
		    if j - 1 = round(mnistArray(i, 0)) then
			target(j) := 1
		    else
			target(j) := 0
		    end if
		end for

		forwardPropagation

		var loss : real := 0
		for j : 1 .. OUTPUT_NODES
		    loss += (target(j) - outputLayer(j)) ** 2
		end for
		batchLoss += loss

		backpropagation(target)
	    end for

	    totalEpochLoss += batchLoss

		cls
		put "Epoch: ", epoch, " / ", EPOCHS
		put "Progress: ", batchEnd, " / ", imageCount
		put "Current Batch Loss: ", batchLoss / (batchEnd - batchStart + 1)
		put "Learning Rate: ", LEARNING_RATE
		View.Update
	end for

	var averageEpochLoss : real := totalEpochLoss / imageCount
	put "Epoch ", epoch, " completed. Average Loss: ", averageEpochLoss

	if averageEpochLoss < bestLoss then
	    bestLoss := averageEpochLoss
	    epochsWithoutImprovement := 0
	    saveModel
	else
	    epochsWithoutImprovement += 1
	    if epochsWithoutImprovement >= EARLY_STOPPING_PATIENCE then
		put "Early stopping triggered. No improvement for ", EARLY_STOPPING_PATIENCE, " epochs."
		exit
	    end if
	end if
    end for
end train

function predict : int
    forwardPropagation
    var maxOutput : real := outputLayer(1)
    var prediction : int := 0
    for i : 2 .. OUTPUT_NODES
	if outputLayer(i) > maxOutput then
	    maxOutput := outputLayer(i)
	    prediction := i - 1
	end if
    end for
    result prediction
end predict

function loadModel : boolean
    var modelFile : int
    open : modelFile, MODEL_FILE, get
    put modelFile
    if not modelFile > 0 then
	result false
    end if
    cls
    View.Update
    put "loading from ", MODEL_FILE
    for i : 1 .. INPUT_NODES
	for j : 1 .. HIDDEN_NODES
	    get : modelFile, weightsIH(i, j)
	end for
    end for

    for i : 1 .. HIDDEN_NODES
	for j : 1 .. OUTPUT_NODES
	    get : modelFile, weightsHO(i, j)
	end for
	get : modelFile, biasH(i)
    end for

    for i : 1 .. OUTPUT_NODES
	get : modelFile, biasO(i)
    end for
    cls
    View.Update
    close : modelFile
    put "model loaded from ", MODEL_FILE
    result true
end loadModel

loadData

var modelLoaded : boolean := loadModel
if not modelLoaded then
    put "no saved model found. initializing weights and training..."
    initializeWeights
    train
else
    put "saved model loaded. skipping training."
end if
cls
View.Update
currentImage := 0
loop
    for y : 0 .. 27
	for x : 0 .. 27
	    pixel := round(mnistArray(currentImage, y * 28 + x + 1) * 255)
	    RGB.SetColour (colourCode, 255 - pixel, 255 - pixel, 255 - pixel)
	    drawfillbox (maxx div 2 - 280 + x * 20,
		maxy div 2 + 280 - y * 20 - 20,
		maxx div 2 - 280 + x * 20 + 20,
		maxy div 2 + 280 - y * 20,
		colourCode)
	end for
    end for
    
    for i : 1 .. INPUT_NODES
	inputLayer(i) := mnistArray(currentImage, i)
    end for
    var prediction : int := predict
    
    Draw.Text ("actual number: " + intstr(round(mnistArray(currentImage, 0))), 
	       maxx div 2 - 100, maxy - 50, fontID, 255)
    if round(mnistArray(currentImage, 0)) = prediction then 
	Draw.Text ("predicted number: " + intstr(prediction),
		   maxx div 2 - 100, maxy - 70, fontID, green)
    else 
	Draw.Text ("predicted number: " + intstr(prediction),
		   maxx div 2 - 100, maxy - 70, fontID, red)
    end if
    Draw.Text ("current image: " + intstr(currentImage + 1) + "/" + intstr(imageCount), 
	       maxx div 2 - 100, maxy - 90, fontID, 255)
    View.Update
    currentChar := getchar
    if ord(currentChar) = 203 and currentImage > 0 then
	currentImage -= 1
    elsif ord(currentChar) = 205 and currentImage < imageCount - 1 then
	currentImage += 1
    end if
    cls
end loop
