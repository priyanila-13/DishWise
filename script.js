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



// Function to handle form submission
function submitForm(event) {
    event.preventDefault(); // Prevent default form submission behavior

    // Get values from the form
    const district = document.getElementById('district').value;
    const commodity = document.getElementById('commodity').value;
    const amount = parseFloat(document.getElementById('amount').value);
    const unit = document.querySelector('input[name="unit"]:checked').value;
    const month = parseInt(document.getElementById('month').value);
    const year = parseInt(document.getElementById('year').value);

    // Validate inputs
    if (!district || !commodity || isNaN(amount) || !unit || isNaN(month) || isNaN(year)) {
        alert('Please fill in all fields correctly.');
        return;
    }

    // Simulate price prediction (replace this with your actual logic/API call)
    const predictedPrice = calculatePredictedPrice(commodity, district, month, year, amount, unit);

    // Update the results section
    displayResults(predictedPrice, district, commodity, amount, unit, month, year);
}

// Function to calculate the predicted price (sample logic)
function calculatePredictedPrice(commodity, district, month, year, amount, unit) {
    // Mock calculation (replace with real calculation logic)
    const basePrice = 3000; // Example base price per quintal
    const pricePerQuintal = unit === 'kg' ? basePrice / 10 : basePrice; // Convert kg to quintal
    const totalPrice = pricePerQuintal * (amount / (unit === 'kg' ? 1000 : 1)); // Convert amount to quintals
    return totalPrice.toFixed(2); // Return total price as string
}

// Function to display the results
function displayResults(predictedPrice, district, commodity, amount, unit, month, year) {
    // Display predicted price
    const priceResult = document.getElementById('price-result');
    priceResult.innerHTML = `<h4>Predicted Price</h4>
                             <p>Predicted price for ${commodity} in ${district} for ${month}/${year}: ₹${predictedPrice} per quintal</p>`;
    
    // Simulate nearby districts data (replace with actual data)
    const nearbyDistricts = [
        { name: 'Erode', distance: 30.12, variety: 'Finger', modalPrice: 2800 },
        { name: 'Karur', distance: 60.75, variety: 'Bulb', modalPrice: 2900 },
        { name: 'Theni', distance: 100.23, variety: 'Other', modalPrice: 2950 },
        { name: 'Namakkal', distance: 150.98, variety: 'Finger', modalPrice: 2700 },
    ];

    // Display nearby district prices
    const locationResult = document.getElementById('location-result');
    locationResult.innerHTML = `<h4>Nearby Districts for ${commodity}</h4>
                                <p>${commodity} is available in the following districts near ${district}:</p>
                                <ul>${nearbyDistricts.map(district => 
                                    `<li>${district.name} (${district.distance} km), Variety: ${district.variety}, Modal Price: ₹${district.modalPrice} per quintal</li>`
                                ).join('')}</ul>`;

    // Display price suggestions
    const suggestionsResult = document.getElementById('suggestions-result');
    suggestionsResult.innerHTML = `<h4>Price Suggestions for ${amount} ${unit}</h4>
                                   <ul>${nearbyDistricts.map(district => 
                                       `<li>${district.name} (Distance: ${district.distance} km), Total Price for ${amount} ${unit}: ₹${(district.modalPrice * (amount / (unit === 'kg' ? 1000 : 1))).toFixed(2)}</li>`
                                   ).join('')}</ul>`;

    // Show the results section
    document.getElementById('results-section').classList.remove('hidden');
}
