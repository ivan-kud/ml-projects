document.addEventListener("DOMContentLoaded", startup);

const text = document.getElementById('text');
const output1 = document.getElementById('output1');
const output2 = document.getElementById('output2');
const output3 = document.getElementById('output3');
const output4 = document.getElementById('output4');
const output5 = document.getElementById('output5');
const output6 = document.getElementById('output6');
const output7 = document.getElementById('output7');
const output8 = document.getElementById('output8');
const output9 = document.getElementById('output9');
const output10 = document.getElementById('output10');
const output11 = document.getElementById('output11');
const output12 = document.getElementById('output12');
const output13 = document.getElementById('output13');
const output14 = document.getElementById('output14');
const output15 = document.getElementById('output15');
const info = document.getElementById('info');

// keep scroll position while refreshing the page
window.onbeforeunload = function(e) {
    localStorage.setItem('scrollPosition', window.scrollY);
};

function startup() {
    // restore scroll position
    var scrollPosition = localStorage.getItem('scrollPosition');
    if (scrollPosition) window.scrollTo(0, scrollPosition);
}

function submitForm() {
    info.innerHTML = '&nbsp';
    output2.innerHTML = '&nbsp';
    output3.innerHTML = '&nbsp';
    output4.innerHTML = '&nbsp';
    output5.innerHTML = '&nbsp';
    output6.innerHTML = '&nbsp';
    output7.innerHTML = '&nbsp';
    output8.innerHTML = '&nbsp';
    output9.innerHTML = '&nbsp';
    output10.innerHTML = '&nbsp';
    output11.innerHTML = '&nbsp';
    output12.innerHTML = '&nbsp';
    output13.innerHTML = '&nbsp';
    output14.innerHTML = '&nbsp';
    output15.innerHTML = '&nbsp';
    if (text.value < 1) {
        output1.innerHTML = 'Write a text please.';
    }
    else {
        output1.innerHTML = 'Processing… Please wait.';
        document.getElementById('myForm').submit();
        document.getElementById('submitBtn').disabled = true;
    }
}
