package com.example.neural

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
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
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.input.pointer.pointerInput
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

//training
import org.jetbrains.kotlinx.dl.dataset.mnist

const val DEBUG_MODE = 1

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val scope = CoroutineScope(Dispatchers.IO)

        val inputSize = 28 * 28
        val hiddenSize = 300
        val hidden2size = 180
        val outputSize = 10
        val neural = SimpleNeuralNetwork(inputSize, hiddenSize, hidden2size, outputSize, useSavedModel = true, context = applicationContext )
        /*scope.launch {
            withContext(Dispatchers.IO) {
                train(neural)
            }
        }*/
        setContent {
            MainScreen(neural, applicationContext)
        }
    }

    private fun train(neural: SimpleNeuralNetwork){
        if(!neural.actionAllow)return
        //массив данных обучения
        val imagesList: Array<DoubleArray> = Array(10000) { DoubleArray(0) }
        val valuesList: Array<DoubleArray> = Array(10000) { DoubleArray(0) }
        val datasList: Array<Int> = Array<Int>(10){0}

        // Загрузка датасета MNIST
        val cacheDir = this.cacheDir
        val (train, test) = mnist(cacheDirectory = cacheDir)

        for (i in 0..9999) {
            //извлечени изображения из обучающего набора
            val firstImage = train.x[i]
            val firstImageLabel = train.y[i]

            datasList[firstImageLabel.toInt()]++

            imagesList[i] = DoubleArray(28 * 28)

            val numClasses = 10
            valuesList[i] = DoubleArray(numClasses) { 0.0 }
            valuesList[i][firstImageLabel.toInt()] = 1.0
            println("init value ${firstImageLabel.toInt()}")

            for (y in 0 until 28) {
                for (x in 0 until 28) {
                    // Извлечение пиксельного значения и преобразование его в диапазон 0-255
                    val pixelValue = (firstImage[y * 28 + x] * 255).toDouble()
                    imagesList[i][y * 28 + x] = pixelValue
                }
            }
        }
        print("startTraaaaain "+datasList.joinToString(separator = " | "))
        neural.train(imagesList, valuesList, baseContext)
    }
}

fun cropEmptyPixels(bitmap: Bitmap): Bitmap {
    val width = bitmap.width
    val height = bitmap.height

    var minX = width
    var minY = height
    var maxX = 0
    var maxY = 0

    // Найдем границы области с содержимым
    for (y in 0 until height) {
        for (x in 0 until width) {
            if (bitmap.getPixel(x, y) != android.graphics.Color.TRANSPARENT) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }

    // Рассчитаем размеры квадрата, содержащего все непустые пиксели
    val squareSize = maxOf(maxX - minX + 1, maxY - minY + 1)+40

    // Создадим новое изображение с новыми размерами
    val croppedBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)

    // Создаем холст для нового изображения
    val canvas = android.graphics.Canvas(croppedBitmap)

    // Рассчитаем координаты для копирования области с содержимым в новое изображение
    val offsetX = ((squareSize - (maxX - minX + 1)) / 2)
    val offsetY = ((squareSize - (maxY - minY + 1)) / 2)

    // Копируем область с содержимым в новое изображение
    val srcRect = Rect(minX, minY, maxX + 1, maxY + 1)
    val destRect = Rect(offsetX, offsetY, offsetX + maxX - minX + 1, offsetY + maxY - minY + 1)
    canvas.drawBitmap(bitmap, srcRect, destRect, null)

    return croppedBitmap
}


@Composable
fun MainScreen(neural: SimpleNeuralNetwork?, context: Context?) {
    var showDialog by remember { mutableStateOf(false) }
    var dialogBitmap by remember { mutableStateOf<ImageBitmap?>(null) }
    var pixels = IntArray(28 * 28)
    var canvasWidth: Int = 0
    var canvasHeight: Int = 0

    var lines by remember { mutableStateOf(emptyList<Pair<Offset, Offset>>()) }
    var rememberedNumber by remember { mutableStateOf<Int?>(0) }
    var recognizedNumber by remember { mutableStateOf<Int?>(null) }
    var isDropdownMenuExpanded by remember { mutableStateOf(false) }
    val dropdownItems = (0..9).toList()
    val padding = 16.dp // Отступ для области холста
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    @Composable
    fun MyDialogWithImage(
        onDismiss: () -> Unit
    ) {
        if (showDialog) {
            AlertDialog(
                onDismissRequest = { showDialog = false; onDismiss()},
                title = { Text(text = "Подтверждение действия") },
                text = {Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Box(
                        modifier = Modifier.background(color= Color.Gray)
                    ) {
                        Image(
                            bitmap = dialogBitmap!!,
                            contentDescription = null,
                            contentScale = ContentScale.Inside,
                        )
                    }
                }
                },
                confirmButton = {
                    Button(onClick = { showDialog = false; onDismiss() },) {
                        Text("OK")
                    }
                }
            )
        }
    }

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
                .border(1.dp, Color.Black)
                .padding(padding)
                .fillMaxWidth()
                .pointerInput(Unit) {
                    detectDragGestures(
                        onDragStart = {
                            lines = lines + Pair(it, it)
                        },
                        onDrag = { change, _ ->
                            val position = change.position
                            if (position.x.toInt() in 0..size.width && position.y.toInt() in 0..size.height) {
                                lines =
                                    lines + Pair(lines.lastOrNull()?.second ?: position, position)
                            }
                        }
                    )
                }
        ) {
            canvasWidth = size.width.toInt()
            canvasHeight = size.height.toInt()
            drawIntoCanvas {
                lines.forEach { (start, end) ->
                    drawLine(
                        color = Color.Black,
                        start = start,
                        end = end,
                        strokeWidth = 50f
                    )
                }
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Divider(
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier
                    .height(40.dp)  //fill the max height
                    .width(1.dp)
            )
            Button(contentPadding = PaddingValues(horizontal = 6.dp), onClick = {
                // Создаем Bitmap и рисуем на нем
                val tempBitmap = Bitmap.createBitmap(canvasWidth, canvasHeight, Bitmap.Config.ARGB_8888)
                val canvas = android.graphics.Canvas(tempBitmap)
                val paint = Paint().apply {
                    color = android.graphics.Color.BLACK
                    strokeWidth = 40f
                }
                lines.forEach { (start, end) ->
                    canvas.drawLine(start.x, start.y, end.x, end.y, paint)
                }

                // Обрезаем пустые пиксели по краям
                val croppedBitmap = cropEmptyPixels(tempBitmap)

                // Масштабируем Bitmap до 28x28
                val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, 28, 28, true)
                scaledBitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28)

                for (i in pixels.indices) {
                    if (pixels[i] != 0) {
                        pixels[i] = 255
                    }
                }
                dialogBitmap = croppedBitmap.asImageBitmap()
                if(DEBUG_MODE==1)showDialog = true

                val images = Array<DoubleArray>(10){ DoubleArray(28*28) }
                val value = arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                if(rememberedNumber!=null)value[rememberedNumber!!]=1.0
                val valueArray = Array<DoubleArray>(10){ DoubleArray(28*28) }
                for (i in 0..9){
                    images[i]=pixels.map { it.toDouble() }.toDoubleArray()
                    valueArray[i]=value.toDoubleArray()
                }
                println("train value - $rememberedNumber for current image")
                neural?.train(images, valueArray, context!!)
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
            Divider(
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier
                    .height(40.dp)  //fill the max height
                    .width(1.dp)
            )
            Button(onClick = {
                // Создаем Bitmap и рисуем на нем
                val tempBitmap = Bitmap.createBitmap(canvasWidth, canvasHeight, Bitmap.Config.ARGB_8888)
                val canvas = android.graphics.Canvas(tempBitmap)
                val paint = Paint().apply {
                    color = android.graphics.Color.BLACK
                    strokeWidth = 40f
                }
                lines.forEach { (start, end) ->
                    canvas.drawLine(start.x, start.y, end.x, end.y, paint)
                }

                // Обрезаем пустые пиксели по краям
                val croppedBitmap = cropEmptyPixels(tempBitmap)

                dialogBitmap = croppedBitmap.asImageBitmap()
                if(DEBUG_MODE==1)showDialog = true
                // Масштабируем Bitmap до 28x28
                val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, 28, 28, true)
                scaledBitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28)

                for (i in pixels.indices) {
                    if (pixels[i] != 0) {
                        pixels[i] = 255
                    }
                }
                dialogBitmap = croppedBitmap.asImageBitmap()
                if(DEBUG_MODE==1)showDialog = true


                val result = if(neural?.actionAllow==true)(neural.predict(pixels.map { it.toDouble() }.toDoubleArray())) else arrayOf(1.0)
                val intResult = result.indexOf(result.max())
                recognizedNumber = intResult
                println("reconition result - $intResult")
            }) {
                Text("Распознать")
            }
            Box(modifier = Modifier
                .height(40.dp)
                .width(40.dp)
                .clip(CircleShape)
                .background(color = MaterialTheme.colorScheme.primary)) {
                BasicTextField(
                    value = recognizedNumber?.toString() ?: "",
                    onValueChange = { },
                    readOnly = true,
                    textStyle = TextStyle(
                        color = Color.White,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center
                    ),
                    modifier = Modifier
                        .align(Alignment.Center)
                        .fillMaxWidth()
                )
            }
            Divider(
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier
                    .height(40.dp)  //fill the max height
                    .width(1.dp)
            )
        }
        MyDialogWithImage(onDismiss = { /* Обработка закрытия диалога */ })
    }
}

@Preview
@Composable
fun PreviewMainScreen() {
    MainScreen(null, null)
}