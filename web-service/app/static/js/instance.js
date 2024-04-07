document.addEventListener("DOMContentLoaded", startup);

const fileBtn = document.getElementById('fileBtn');
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
    if (fileBtn.value == null || fileBtn.value == "") {
        info.innerHTML = 'Choose a file.';
    } else {
        info.innerHTML = 'Processing… Please wait.';
        document.getElementById('myForm').submit();
        fileBtn.disabled = true;
        document.getElementById('submitBtn').disabled = true;
    }
}
