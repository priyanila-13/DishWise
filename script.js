document.addEventListener('DOMContentLoaded', function () {
    const slides = document.querySelectorAll('.slide');
    let currentSlide = 0;

    function changeSlide() {
        slides.forEach(slide => slide.classList.remove('slide-active'));
        slides[currentSlide].classList.add('slide-active');
        currentSlide = (currentSlide + 1) % slides.length;
    }

    changeSlide(); // Set first slide as active initially
    setInterval(changeSlide, 6000); // Change slide every 6 seconds
});


window.addEventListener("scroll", function() {
    // Get the slider element
    const slider = document.querySelector(".slider");
    // Get the height of the slider
    const sliderHeight = slider.offsetHeight;
    // Get the current scroll position
    const scrollPosition = window.scrollY;
    // Get all text elements inside the slider
    const textOverlayElements = document.querySelectorAll(".text-overlay");

    // Apply blur effect based on scroll position
    if (scrollPosition < sliderHeight) {
        const blurValue = (scrollPosition / sliderHeight) * 10; // Adjust the blur intensity
        textOverlayElements.forEach((element) => {
            element.style.filter = `blur(${blurValue}px)`;
        });
    } else {
        // Remove blur when scrolled past the slider
        textOverlayElements.forEach((element) => {
            element.style.filter = "none";
        });
    }
});


function submitForm(event) {
    event.preventDefault();

    // Get form data
    const district = document.getElementById('district').value;
    const commodity = document.getElementById('commodity').value;
    const variety = document.getElementById('variety').value;
    const amount = document.getElementById('amount').value;
    const month = document.getElementById('month').value;
    const year = document.getElementById('year').value;

    // Prepare the data to send
    const data = {
        district: district,
        commodity: commodity,
        variety: variety,
        amount: amount,
        month: month,
        year: year
    };

    // Send the data to the Flask app
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            console.error(result.error);
            alert('Error: ' + result.error);
            return;
        }

        // Populate predicted price
        document.getElementById('predicted-price').textContent = 
            `Predicted price for ${commodity} in ${district} for ${month}/${year}: ₹${result.predictedPrice.toFixed(2)} per quintal`;

        // Populate nearby districts
        const locationList = document.getElementById('location-list');
        locationList.innerHTML = ''; // Clear previous results
        result.nearbyDistricts.forEach(district => {
            const li = document.createElement('li');
            li.textContent = `${district.name} (${district.distance.toFixed(2)} km), Price: ₹${district.price.toFixed(2)} for ${amount} kg`;
            locationList.appendChild(li);
        });

        // Show best buy month
        document.getElementById('best-buy-month').textContent = 
            `Best buy month for ${commodity} (${variety}) is ${result.bestBuyMonth}.`;

        // Update amount display
        document.getElementById('amount-display').textContent = `${amount} kg`;

        // Show results section
        document.getElementById('results-section').classList.remove('hidden');
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    });
}
