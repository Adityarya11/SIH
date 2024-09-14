document.getElementById('submitDate').addEventListener('click', function () {
    var selectedDate = document.getElementById('dateInput').value;
    if (!selectedDate) {
        document.getElementById('resultDiv').textContent = 'Please select a valid date.';
        return;
    }

    var dateParts = selectedDate.split("-");
    var year = parseInt(dateParts[0]);
    var month = parseInt(dateParts[1]);
    var day = parseInt(dateParts[2]);

    fetch('/api/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({
            'selectDate': day,
            'selectMonth': month,
            'selectYear': year
        })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('resultDiv');
        resultDiv.innerHTML = '';  // Clear previous results

        if (data.predictions) {
            for (let time in data.predictions) {
                const load = data.predictions[time];
                const predictionDiv = document.createElement('div');
                predictionDiv.textContent = `${time}: ${load.toFixed(2)} MW`;
                resultDiv.appendChild(predictionDiv);
            }
        } else if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('resultDiv').textContent = 'An error occurred while fetching the prediction.';
    });
});

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
