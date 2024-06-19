package com.example.neural

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.math.exp
import kotlin.random.Random

class SimpleNeuralNetwork(
    private val inputSize: Int,
    private val hiddenSize1: Int,
    private val hiddenSize2: Int,
    private val outputSize: Int,
    private val learningRate: Double = 0.01,
    private val useSavedModel: Boolean = false,
    private val context: Context
) {
    private val inputLayer: Array<Double>
    private val hiddenLayer1: Array<Double>
    private val hiddenLayer2: Array<Double>
    private val outputLayer: Array<Double>

    private var weightsInputHidden1: Array<Array<Double>>
    private var weightsHidden1Hidden2: Array<Array<Double>>
    private var weightsHidden2Output: Array<Array<Double>>

    private val biasHidden1: DoubleArray
    private val biasHidden2: DoubleArray
    private val biasOutput: DoubleArray

    public var actionAllow = false

    init {
        inputLayer = DoubleArray(inputSize).toTypedArray()
        hiddenLayer1 = DoubleArray(hiddenSize1).toTypedArray()
        hiddenLayer2 = DoubleArray(hiddenSize2).toTypedArray()
        outputLayer = DoubleArray(outputSize).toTypedArray()

        weightsInputHidden1 =
            Array(hiddenSize1) { Array(inputSize) { Random.nextDouble(-1.0, 1.0) } }
        weightsHidden1Hidden2 =
            Array(hiddenSize2) { Array(hiddenSize1) { Random.nextDouble(-1.0, 1.0) } }
        weightsHidden2Output =
            Array(outputSize) { Array(hiddenSize2) { Random.nextDouble(-1.0, 1.0) } }

        biasHidden1 = DoubleArray(hiddenSize1) { Random.nextDouble(-1.0, 1.0) }
        biasHidden2 = DoubleArray(hiddenSize2) { Random.nextDouble(-1.0, 1.0) }
        biasOutput = DoubleArray(outputSize) { Random.nextDouble(-1.0, 1.0) }

        if (useSavedModel) {
            loadModel(context)
        }
    }

    fun train(inputData: Array<DoubleArray>, targets: Array<DoubleArray>, context: Context) {
        require(inputData.size == targets.size) { "Количество входных данных должно быть равно количеству целевых значений" }

        for (i in inputData.indices) {

            println("step - $i value - ${targets[i].indexOfFirst { it==1.0 }}")

            val input = inputData[i]
            val target = targets[i]

            // Forward pass
            for (j in input.indices) {
                inputLayer[j] = input[j]
            }

            // Forward pass through first hidden layer
            for (j in hiddenLayer1.indices) {
                var sum = 0.0
                for (k in inputLayer.indices) {
                    sum += inputLayer[k] * weightsInputHidden1[j][k]
                }
                hiddenLayer1[j] = sigmoid(sum + biasHidden1[j])
            }

            // Forward pass through second hidden layer
            for (j in hiddenLayer2.indices) {
                var sum = 0.0
                for (k in hiddenLayer1.indices) {
                    sum += hiddenLayer1[k] * weightsHidden1Hidden2[j][k]
                }
                hiddenLayer2[j] = sigmoid(sum + biasHidden2[j])
            }

            // Forward pass through output layer
            for (j in outputLayer.indices) {
                var sum = 0.0
                for (k in hiddenLayer2.indices) {
                    sum += hiddenLayer2[k] * weightsHidden2Output[j][k]
                }
                outputLayer[j] = sigmoid(sum + biasOutput[j])
            }

            // Backpropagation
            val outputErrors = DoubleArray(outputSize)
            for (j in outputLayer.indices) {
                val error = target[j] - outputLayer[j]
                outputErrors[j] = error * sigmoidDerivative(outputLayer[j])
            }

            val hiddenErrors2 = DoubleArray(hiddenSize2)
            for (j in hiddenLayer2.indices) {
                var error = 0.0
                for (k in outputLayer.indices) {
                    error += outputErrors[k] * weightsHidden2Output[k][j]
                }
                hiddenErrors2[j] = error * sigmoidDerivative(hiddenLayer2[j])
            }

            val hiddenErrors1 = DoubleArray(hiddenSize1)
            for (j in hiddenLayer1.indices) {
                var error = 0.0
                for (k in hiddenLayer2.indices) {
                    error += hiddenErrors2[k] * weightsHidden1Hidden2[k][j]
                }
                hiddenErrors1[j] = error * sigmoidDerivative(hiddenLayer1[j])
            }

            for (j in outputLayer.indices) {
                for (k in hiddenLayer2.indices) {
                    weightsHidden2Output[j][k] += learningRate * outputErrors[j] * hiddenLayer2[k]
                }
                biasOutput[j] += learningRate * outputErrors[j]
            }

            for (j in hiddenLayer2.indices) {
                for (k in hiddenLayer1.indices) {
                    weightsHidden1Hidden2[j][k] += learningRate * hiddenErrors2[j] * hiddenLayer1[k]
                }
                biasHidden2[j] += learningRate * hiddenErrors2[j]
            }

            for (j in hiddenLayer1.indices) {
                for (k in inputLayer.indices) {
                    weightsInputHidden1[j][k] += learningRate * hiddenErrors1[j] * inputLayer[k]
                }
                biasHidden1[j] += learningRate * hiddenErrors1[j]
            }
        }

        println("save model for ${inputData.size} items")
        saveModel(context)
    }

    fun predict(inputData: DoubleArray): Array<Double> {
        for (i in inputData.indices) {
            inputLayer[i] = inputData[i]
        }

        // Forward pass through first hidden layer
        for (i in hiddenLayer1.indices) {
            var sum = 0.0
            for (j in inputLayer.indices) {
                sum += inputLayer[j] * weightsInputHidden1[i][j]
            }
            hiddenLayer1[i] = sigmoid(sum + biasHidden1[i])
        }

        // Forward pass through second hidden layer
        for (i in hiddenLayer2.indices) {
            var sum = 0.0
            for (j in hiddenLayer1.indices) {
                sum += hiddenLayer1[j] * weightsHidden1Hidden2[i][j]
            }
            hiddenLayer2[i] = sigmoid(sum + biasHidden2[i])
        }

        // Forward pass through output layer
        for (i in outputLayer.indices) {
            var sum = 0.0
            for (j in hiddenLayer2.indices) {
                sum += hiddenLayer2[j] * weightsHidden2Output[i][j]
            }
            outputLayer[i] = sigmoid(sum + biasOutput[i])
        }

        return outputLayer
    }

    private fun saveModel(context: Context) {
        val sharedPreferences: SharedPreferences = context.getSharedPreferences("NeuralNetworkPrefs", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()

        // Сохранение размеров
        editor.putInt("inputSize", inputSize)
        editor.putInt("hiddenSize1", hiddenSize1)
        editor.putInt("hiddenSize2", hiddenSize2)
        editor.putInt("outputSize", outputSize)

        // Сохранение весов первого скрытого слоя
        for (i in 0 until hiddenSize1) {
            for (j in 0 until inputSize) {
                editor.putFloat("weightsInputHidden1_$i$j", weightsInputHidden1[i][j].toFloat())
            }
        }

        // Сохранение весов второго скрытого слоя
        for (i in 0 until hiddenSize2) {
            for (j in 0 until hiddenSize1) {
                editor.putFloat("weightsHidden1Hidden2_$i$j", weightsHidden1Hidden2[i][j].toFloat())
            }
        }

        // Сохранение весов выходного слоя
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize2) {
                editor.putFloat("weightsHiddenOutput_$i$j", weightsHidden2Output[i][j].toFloat())
            }
        }

        // Сохранение смещений первого скрытого слоя
        for (i in 0 until hiddenSize1) {
            editor.putFloat("biasHidden1_$i", biasHidden1[i].toFloat())
        }

        // Сохранение смещений второго скрытого слоя
        for (i in 0 until hiddenSize2) {
            editor.putFloat("biasHidden2_$i", biasHidden2[i].toFloat())
        }

        // Сохранение смещений выходного слоя
        for (i in 0 until outputSize) {
            editor.putFloat("biasOutput_$i", biasOutput[i].toFloat())
        }

        editor.apply()
    }

    private fun loadModel(context: Context) {
        CoroutineScope(Dispatchers.IO).launch {
            val sharedPreferences: SharedPreferences =
                context.getSharedPreferences("NeuralNetworkPrefs", Context.MODE_PRIVATE)

            // Загрузка размеров
            val savedInputSize = sharedPreferences.getInt("inputSize", -1)
            val savedHiddenSize1 = sharedPreferences.getInt("hiddenSize1", -1)
            val savedHiddenSize2 = sharedPreferences.getInt("hiddenSize2", -1)
            val savedOutputSize = sharedPreferences.getInt("outputSize", -1)
            println("Loaded model sizes: inputSize=$savedInputSize, hiddenSize1=$savedHiddenSize1, hiddenSize2=$savedHiddenSize2, outputSize=$savedOutputSize")

            if (savedInputSize != -1 && savedHiddenSize1 != -1 && savedHiddenSize2 != -1 && savedOutputSize != -1) {
                for (i in 0 until hiddenSize1) {
                    for (j in 0 until inputSize) {
                        weightsInputHidden1[i][j] =
                            sharedPreferences.getFloat("weightsInputHidden1_$i$j", 0.0f).toDouble()
                    }
                }

                for (i in 0 until hiddenSize2) {
                    for (j in 0 until hiddenSize1) {
                        weightsHidden1Hidden2[i][j] =
                            sharedPreferences.getFloat("weightsHidden1Hidden2_$i$j", 0.0f)
                                .toDouble()
                    }
                }

                for (i in 0 until outputSize) {
                    for (j in 0 until hiddenSize2) {
                        weightsHidden2Output[i][j] =
                            sharedPreferences.getFloat("weightsHiddenOutput_$i$j", 0.0f).toDouble()
                    }
                }

                for (i in 0 until hiddenSize1) {
                    biasHidden1[i] = sharedPreferences.getFloat("biasHidden1_$i", 0.0f).toDouble()
                }

                for (i in 0 until hiddenSize2) {
                    biasHidden2[i] = sharedPreferences.getFloat("biasHidden2_$i", 0.0f).toDouble()
                }

                for (i in 0 until outputSize) {
                    biasOutput[i] = sharedPreferences.getFloat("biasOutput_$i", 0.0f).toDouble()
                }
            } else {
                println("Saved model not found or sizes do not match.")
            }
            println("model loaded")
        }
        actionAllow =true;
    }

    private fun sigmoid(x: Double): Double {
        return 1 / (1 + exp(-x))
    }

    private fun sigmoidDerivative(x: Double): Double {
        val sig = sigmoid(x)
        return sig * (1 - sig)
    }
}

/*class SimpleNeuralNetwork(
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val outputSize: Int,
    private val learningRate: Double = 0.01,
    private val useSavedModel: Boolean = false,
    private val context: Context
) {
    private val inputLayer: Array<Double>
    private val hiddenLayer: Array<Double>
    private val outputLayer: Array<Double>

    private var weightsInputHidden: Array<Array<Double>>
    private var weightsHiddenOutput: Array<Array<Double>>

    private val biasHidden: DoubleArray
    private val biasOutput: DoubleArray

    init {
        inputLayer = DoubleArray(inputSize).toTypedArray()
        hiddenLayer = DoubleArray(hiddenSize).toTypedArray()
        outputLayer = DoubleArray(outputSize).toTypedArray()

        weightsInputHidden = Array(hiddenSize) { Array(inputSize) { Random.nextDouble(-1.0, 1.0) } }
        weightsHiddenOutput = Array(outputSize) { Array(hiddenSize) { Random.nextDouble(-1.0, 1.0) } }

        biasHidden = DoubleArray(hiddenSize) { Random.nextDouble(-1.0, 1.0) }
        biasOutput = DoubleArray(outputSize) { Random.nextDouble(-1.0, 1.0) }

        if (useSavedModel){
            loadModel(context)
        }
    }

    fun train(inputData: Array<DoubleArray>, targets: Array<DoubleArray>, context: Context) {
        require(inputData.size == targets.size) { "Количество входных данных должно быть равно количеству целевых значений" }

        for (i in inputData.indices) {

            var str = ""
            targets[i].forEach {
                str+="$it,"
            }
            println("step - $i value - $str")

            val input = inputData[i]
            val target = targets[i]

            // Forward pass
            for (j in input.indices) {
                inputLayer[j] = input[j]
            }

            for (j in hiddenLayer.indices) {
                var sum = 0.0
                for (k in inputLayer.indices) {
                    sum += inputLayer[k] * weightsInputHidden[j][k]
                }
                hiddenLayer[j] = sigmoid(sum + biasHidden[j])
            }

            for (j in outputLayer.indices) {
                var sum = 0.0
                for (k in hiddenLayer.indices) {
                    sum += hiddenLayer[k] * weightsHiddenOutput[j][k]
                }
                outputLayer[j] = sigmoid(sum + biasOutput[j])
            }

            // Backpropagation
            val outputErrors = DoubleArray(outputSize)
            for (j in outputLayer.indices) {
                val error = target[j] - outputLayer[j]
                outputErrors[j] = error * sigmoidDerivative(outputLayer[j])
            }

            val hiddenErrors = DoubleArray(hiddenSize)
            for (j in hiddenLayer.indices) {
                var error = 0.0
                for (k in outputLayer.indices) {
                    error += outputErrors[k] * weightsHiddenOutput[k][j]
                }
                hiddenErrors[j] = error * sigmoidDerivative(hiddenLayer[j])
            }

            for (j in outputLayer.indices) {
                for (k in hiddenLayer.indices) {
                    weightsHiddenOutput[j][k] += learningRate * outputErrors[j] * hiddenLayer[k]
                }
                biasOutput[j] += learningRate * outputErrors[j]
            }

            for (j in hiddenLayer.indices) {
                for (k in inputLayer.indices) {
                    weightsInputHidden[j][k] += learningRate * hiddenErrors[j] * inputLayer[k]
                }
                biasHidden[j] += learningRate * hiddenErrors[j]
            }
        }

        println("save model for ${inputData.size} items")
        saveModel(context)
    }


    fun predict(inputData: DoubleArray): Array<Double> {
        for (i in inputData.indices) {
            inputLayer[i] = inputData[i]
        }

        for (i in hiddenLayer.indices) {
            var sum = 0.0
            for (j in inputLayer.indices) {
                sum += inputLayer[j] * weightsInputHidden[i][j]
            }
            hiddenLayer[i] = sigmoid(sum + biasHidden[i])
        }

        for (i in outputLayer.indices) {
            var sum = 0.0
            for (j in hiddenLayer.indices) {
                sum += hiddenLayer[j] * weightsHiddenOutput[i][j]
            }
            outputLayer[i] = sigmoid(sum + biasOutput[i])
        }

        return outputLayer
    }

    fun saveModel(context: Context) {
        val sharedPreferences: SharedPreferences = context.getSharedPreferences("NeuralNetworkPrefs", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()

        // Сохранение весов и смещений
        editor.putInt("inputSize", inputSize)
        editor.putInt("hiddenSize", hiddenSize)
        editor.putInt("outputSize", outputSize)
        // Сохранение весов
        for (i in 0 until hiddenSize) {
            for (j in 0 until inputSize) {
                editor.putFloat("weightsInputHidden_$i$j", weightsInputHidden[i][j].toFloat())
            }
        }
        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                editor.putFloat("weightsHiddenOutput_$i$j", weightsHiddenOutput[i][j].toFloat())
            }
        }
        // Сохранение смещений
        for (i in 0 until hiddenSize) {
            editor.putFloat("biasHidden_$i", biasHidden[i].toFloat())
        }
        for (i in 0 until outputSize) {
            editor.putFloat("biasOutput_$i", biasOutput[i].toFloat())
        }

        editor.apply()
    }

    private fun loadModel(context: Context) {
        val sharedPreferences: SharedPreferences = context.getSharedPreferences("NeuralNetworkPrefs", Context.MODE_PRIVATE)

        // Загрузка размеров
        val savedInputSize = sharedPreferences.getInt("inputSize", -1)
        val savedHiddenSize = sharedPreferences.getInt("hiddenSize", -1)
        val savedOutputSize = sharedPreferences.getInt("outputSize", -1)
        println("ggggggggggggggggg $savedInputSize $savedHiddenSize $savedOutputSize")

        if (savedInputSize != -1 && savedHiddenSize != -1 && savedOutputSize != -1) {
            for (i in 0 until hiddenSize) {
                for (j in 0 until inputSize) {
                    weightsInputHidden[i][j] = sharedPreferences.getFloat("weightsInputHidden_$i$j", 0.0f).toDouble()
                }
            }
            for (i in 0 until outputSize) {
                for (j in 0 until hiddenSize) {
                    weightsHiddenOutput[i][j] = sharedPreferences.getFloat("weightsHiddenOutput_$i$j", 0.0f).toDouble()
                }
            }

            // Загрузка смещений
            for (i in 0 until hiddenSize) {
                biasHidden[i] = sharedPreferences.getFloat("biasHidden_$i", 0.0f).toDouble()
            }
            for (i in 0 until outputSize) {
                biasOutput[i] = sharedPreferences.getFloat("biasOutput_$i", 0.0f).toDouble()
            }
        } else {

        }
    }

    private fun sigmoid(x: Double): Double {
        return 1 / (1 + exp(-x))
    }

    private fun sigmoidDerivative(x: Double): Double {
        val sig = sigmoid(x)
        return sig * (1 - sig)
    }
}*/