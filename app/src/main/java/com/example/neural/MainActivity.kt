package com.example.neural

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import com.example.neural.ui.theme.NeuralTheme

class MainActivity : ComponentActivity() {

    /*class OurNeuralNetwork {
        private var w1 = Random.nextDouble()
        private var w2 = Random.nextDouble()
        private var w3 = Random.nextDouble()
        private var w4 = Random.nextDouble()
        private var w5 = Random.nextDouble()
        private var w6 = Random.nextDouble()

        private var b1 = Random.nextDouble()
        private var b2 = Random.nextDouble()
        private var b3 = Random.nextDouble()

        fun sigmoid(x: Double): Double {
            // Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
            return 1 / (1 + exp(-x))
        }

        fun derivSigmoid(x: Double): Double {
            // Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
            val fx = sigmoid(x)
            return fx * (1 - fx)
        }

        fun mseLoss(yTrue: List<Double>, yPred: List<Double>): Double {
            // yTrue and yPred are lists of the same length.
            val n = yTrue.size
            var sum = 0.0
            for (i in yTrue.indices) {
                sum += (yTrue[i] - yPred[i]).pow(2)
            }
            return sum / n
        }

        fun feedforward(x: List<Double>): Double {
            val sumH1 = w1 * x[0] + w2 * x[1] + b1
            val h1 = sigmoid(sumH1)

            val sumH2 = w3 * x[0] + w4 * x[1] + b2
            val h2 = sigmoid(sumH2)

            val sumO1 = w5 * h1 + w6 * h2 + b3
            return sigmoid(sumO1)
        }

        fun train(data: List<List<Double>>, allYTrues: List<Double>, learnRate: Double = 0.1, epochs: Int = 500) {
            for (epoch in 0 until epochs) {
                for (i in data.indices) {
                    val x = data[i]
                    val yTrue = allYTrues[i]

                    // Feedforward
                    val sumH1 = w1 * x[0] + w2 * x[1] + b1
                    val h1 = sigmoid(sumH1)

                    val sumH2 = w3 * x[0] + w4 * x[1] + b2
                    val h2 = sigmoid(sumH2)

                    val sumO1 = w5 * h1 + w6 * h2 + b3
                    val yPred = sigmoid(sumO1)

                    // Backpropagation
                    val dLdYpred = -2 * (yTrue - yPred)
                    val dYpreddW5 = h1 * derivSigmoid(sumO1)
                    val dYpreddW6 = h2 * derivSigmoid(sumO1)
                    val dYpreddB3 = derivSigmoid(sumO1)
                    val dYpreddH1 = w5 * derivSigmoid(sumO1)
                    val dYpreddH2 = w6 * derivSigmoid(sumO1)

                    val dH1dW1 = x[0] * derivSigmoid(sumH1)
                    val dH1dW2 = x[1] * derivSigmoid(sumH1)
                    val dH1dB1 = derivSigmoid(sumH1)

                    val dH2dW3 = x[0] * derivSigmoid(sumH2)
                    val dH2dW4 = x[1] * derivSigmoid(sumH2)
                    val dH2dB2 = derivSigmoid(sumH2)

                    // Update weights and biases
                    w1 -= learnRate * dLdYpred * dYpreddH1 * dH1dW1
                    w2 -= learnRate * dLdYpred * dYpreddH1 * dH1dW2
                    b1 -= learnRate * dLdYpred * dYpreddH1 * dH1dB1

                    w3 -= learnRate * dLdYpred * dYpreddH2 * dH2dW3
                    w4 -= learnRate * dLdYpred * dYpreddH2 * dH2dW4
                    b2 -= learnRate * dLdYpred * dYpreddH2 * dH2dB2

                    w5 -= learnRate * dLdYpred * dYpreddW5
                    w6 -= learnRate * dLdYpred * dYpreddW6
                    b3 -= learnRate * dLdYpred * dYpreddB3
                }

                // Calculate and print total loss at the end of each epoch
                if (epoch % 10 == 0) {
                    val yPreds = data.map { feedforward(it) }
                    val loss = mseLoss(allYTrues, yPreds)
                    println("Epoch $epoch loss: %.3f".format(loss))
                }
            }
        }
    }*/

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        print("staart")
        val data = mapOf(
            Pair(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0), doubleArrayOf(1.0,0.0, 0.0)),
            Pair(doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0), doubleArrayOf(0.0,1.0, 0.0)),
            Pair(doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0), doubleArrayOf(0.0,0.0, 1.0)),
        )
        val inputSize = 5 * 5//28 * 28
        val hiddenSize = 30
        val outputSize = 3
        val neural = SimpleNeuralNetwork(inputSize, hiddenSize, outputSize )
        for (i in 0..10)neural.train(data.keys.toTypedArray(), data.values.toTypedArray())

        val result = neural.predict(data.keys.toTypedArray()[0])
        println("result  -  ${result[0]} - ${result[1]} - ${result[2]}")
        setContent {
            MainScreen()
        }
        // Define dataset
        /*val data = listOf(
                listOf(-2.0, -1.0),  // Alice
                listOf(25.0, 6.0),   // Bob
                listOf(17.0, 4.0),   // Charlie
                listOf(-15.0, -6.0)  // Diana
        )
        val allYTrues = listOf(1.0, 0.0, 0.0, 1.0)

        // Train our neural network
        val network = OurNeuralNetwork()
        network.train(data, allYTrues)

        // Make some predictions
        val emily = listOf(-7.0, -3.0)  // Emily
        val frank = listOf(20.0, 2.0)   // Frank
        println("Emily: %.3f".format(network.feedforward(emily)))  // Expected: ~0.951 (F)
        println("Frank: %.3f".format(network.feedforward(frank)))  // Expected: ~0.039 (M)*/


    }
}

@Composable
fun MainScreen() {
    var lines by remember { mutableStateOf(emptyList<Pair<Offset, Offset>>()) }
    var rememberedNumber by remember { mutableStateOf<Int?>(0) }
    var recognizedNumber by remember { mutableStateOf<Int?>(null) }
    var isDropdownMenuExpanded by remember { mutableStateOf(false) }
    val dropdownItems = (0..9).toList()
    val padding = 16.dp // Отступ для области холста

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Button(onClick = { lines = emptyList() }) {
            Text("Очистить")
        }
        Canvas(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .border(1.dp, Color.Black)
                .padding(padding)
                .pointerInput(Unit) {
                    detectDragGestures(
                        onDragStart = {
                            lines = lines + Pair(it, it)
                        },
                        onDrag = { change, _ ->
                            val position = change.position
                            if (position.x.toInt() in 0..this.size.width && position.y.toInt() in 0..this.size.height
                            ) {
                                lines =
                                    lines + Pair(lines.lastOrNull()?.second ?: position, position)
                            }
                        }
                    )
                }
        ) {
            lines.forEach { (start, end) ->
                drawLine(
                    color = Color.Black,
                    start = start,
                    end = end,
                    strokeWidth = 40f
                )
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(contentPadding = PaddingValues(horizontal = 6.dp), onClick = {
                rememberedNumber = lines.size // For simplicity, using the number of lines as remembered number
            }) {
                Text("Запомнить")
            }
            Box {
                DropdownMenu(
                    expanded = isDropdownMenuExpanded,
                    onDismissRequest = { isDropdownMenuExpanded = false }
                ) {
                    dropdownItems.forEach { number ->
                        DropdownMenuItem(
                            text = { Text(text = number.toString()) },
                            onClick = {
                                rememberedNumber = number
                                isDropdownMenuExpanded = false
                            })
                    }
                }
            }
            Button(onClick = { isDropdownMenuExpanded = true }) {
                Text(rememberedNumber?.toString() ?: "Выберите число")
            }
            Box{}
            Button(onClick = {
                recognizedNumber = lines.size // For simplicity, using the number of lines as recognized number
            }) {
                Text("Распознать")
            }
        }
        TextField(
            value = recognizedNumber?.toString() ?: "",
            onValueChange = { },
            label = { Text("Распознанное число") },
            readOnly = true,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Preview
@Composable
fun PreviewMainScreen() {
    MainScreen()
}