package com.example.neural

import android.content.Context
import android.content.SharedPreferences
import kotlin.math.exp
import kotlin.random.Random

class SimpleNeuralNetwork(
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val outputSize: Int,
    private val learningRate: Double = 0.01
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
    }

    fun train(inputData: Array<DoubleArray>, targets: Array<DoubleArray>) {
        require(inputData.size == targets.size) { "Количество входных данных должно быть равно количеству целевых значений" }

        for (i in inputData.indices) {
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

    private fun saveModel(context: Context) {
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
}
