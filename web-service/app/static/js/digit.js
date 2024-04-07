document.addEventListener("DOMContentLoaded", startup);

const output1 = document.getElementById("output1");
const output2 = document.getElementById("output2");
const image = document.getElementById("image");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
var currCursorPos = { x: 0, y: 0 };
var prevCursorPos = currCursorPos;
var dragging = false;

ctx.lineWidth = 2;
ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.strokeStyle = "black";
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// store canvas data while submitting the form
var tempImage = new Image;
tempImage.onload = function() {
    ctx.drawImage(tempImage, 0, 0);
};
if (image.value != "") tempImage.src = image.value;

// keep scroll position while refreshing the page
window.onbeforeunload = function(e) {
    localStorage.setItem('scrollPosition', window.scrollY);
};

function startup() {
    // set up mouse events for drawing
    canvas.addEventListener("mousedown", e => startDrawing(e.clientX, e.clientY));
    canvas.addEventListener("mousemove", e => drawing(e.clientX, e.clientY));
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseout", stopDrawing);

    // set up touch events for drawing
    canvas.addEventListener("touchstart", function (e) {
        e.preventDefault();
        startDrawing(e.touches[0].clientX, e.touches[0].clientY);
    });
    canvas.addEventListener("touchmove", function (e) {
        e.preventDefault();
        drawing(e.touches[0].clientX, e.touches[0].clientY);
    });
    canvas.addEventListener("touchend", function (e) {
        e.preventDefault();
        stopDrawing();
    });
    canvas.addEventListener("touchcancel", function (e) {
        e.preventDefault();
        stopDrawing();
    });

    // restore scroll position
    var scrollPosition = localStorage.getItem('scrollPosition');
    if (scrollPosition) window.scrollTo(0, scrollPosition);
}

function startDrawing(clientX, clientY) {
    prevCursorPos = getCursorPos(clientX, clientY);
    dragging = true;
}

function drawing(clientX, clientY) {
        if (dragging) {
            currCursorPos = getCursorPos(clientX, clientY);
            draw();
            prevCursorPos = currCursorPos;
        }
}

function stopDrawing() {
    dragging = false;
}

function getCursorPos(clientX, clientY) {
    const canvasBorderWidth = (canvas.offsetWidth - canvas.clientWidth) / 2;
    const canvasBorderHeight = (canvas.offsetHeight - canvas.clientHeight) / 2;
    var rect = canvas.getBoundingClientRect();
    return {
        x: (clientX - rect.left - canvasBorderWidth) * canvas.width / canvas.clientWidth,
        y: (clientY - rect.top - canvasBorderHeight) * canvas.height / canvas.clientHeight
    };
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevCursorPos.x, prevCursorPos.y);
    ctx.lineTo(currCursorPos.x, currCursorPos.y);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    output1.innerHTML = '&nbsp';
    output2.innerHTML = '&nbsp';
}

function submitForm() {
    if (isCanvasBlank()) {
        output1.innerHTML = 'Draw a digit please.';
        output2.innerHTML = '&nbsp';
    } else {
        image.value = canvas.toDataURL();
        document.getElementById('myForm').submit();
        document.getElementById('clearBtn').disabled = true;
        document.getElementById('submitBtn').disabled = true;
    }
}

// returns true if every pixel's uint32 representation is 0xFFFFFFFF
function isCanvasBlank() {
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    return !pixelBuffer.some(color => color !== 0xFFFFFFFF);
}
