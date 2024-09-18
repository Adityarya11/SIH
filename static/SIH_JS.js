function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

document.getElementById('submitDate').addEventListener('click', function () {
    var selectedDate = document.getElementById('dateInput').value;
    var selectedTime = document.getElementById('timeInput').value;

    if (!selectedDate || !selectedTime) {
        document.getElementById('resultDiv').textContent = 'Please select a valid date and time.';
        return;
    }

    fetch('/api/predict/', {  // Add the 'api/' prefix to match your URL structure
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({
            'selectDate': selectedDate,
            'selectTime': selectedTime
        })
    })
    
    
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('resultDiv');
        resultDiv.innerHTML = '';  // Clear previous results

        if (data.prediction) {
            resultDiv.textContent = `Predicted Load: ${data.prediction.toFixed(2)} MW`;
        } else if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('resultDiv').textContent = 'An error occurred while fetching the prediction.';
    });
});
