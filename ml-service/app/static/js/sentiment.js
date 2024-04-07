document.addEventListener("DOMContentLoaded", startup);

const text = document.getElementById('text');
const output1 = document.getElementById('output1');
const output2 = document.getElementById('output2');
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
    if (text.value < 1) {
        output1.innerHTML = 'Write a review please.';
        output2.innerHTML = '&nbsp';
        info.innerHTML = '&nbsp';
    }
    else {
        output1.innerHTML = 'Processing… Please wait.';
        output2.innerHTML = '&nbsp';
        info.innerHTML = '&nbsp';
        document.getElementById('myForm').submit();
        document.getElementById('submitBtn').disabled = true;
    }
}
