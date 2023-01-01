var canvas = document.getElementById("canvas");
var message = document.getElementById("message");

var ctx = canvas.getContext("2d");
ctx.lineWidth = canvas.width / 20;
ctx.lineJoin = "round";
ctx.lineCap = 'round';
ctx.strokeStyle = "black";
ctx.fillStyle = "black";

var prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    flag = false;

canvas.addEventListener("mousemove", e => findxy('move', e), false);
canvas.addEventListener("mousedown", e => findxy('down', e), false);
canvas.addEventListener("mouseup", e => findxy('up', e), false);
canvas.addEventListener("mouseout", e => findxy('out', e), false);

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = getX(e);
        currY = getY(e);
        flag = true;
    }
    if (res == 'up' || res == "out") flag = false;
    if (res == 'move' && flag) {
        prevX = currX;
        prevY = currY;
        currX = getX(e);
        currY = getY(e);
        draw();
    }
}

const getX = (e) => e.clientX - canvas.getBoundingClientRect().left - 1; // 1 is the border width
const getY = (e) => e.clientY - canvas.getBoundingClientRect().top - 1; // 1 is the border width

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    message.innerHTML = '';
}

function submitCanvas() {
    if (isCanvasBlank()) {
        message.innerHTML = '<span style="color:red">Error</span><br>Can\'t send blank image';
    } else {
        document.getElementById('image').value = canvas.toDataURL();
        document.getElementById('myForm').submit();
    }
}

// returns true if every pixel's uint32 representation is 0
function isCanvasBlank() {
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    return !pixelBuffer.some(color => color !== 0);
}