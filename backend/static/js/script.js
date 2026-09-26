function showWebsiteScanner() {
    const websiteScanner = document.getElementById("website-scanner");
    const emailScanner = document.getElementById("email-scanner");

    if (websiteScanner) {
        websiteScanner.style.display = "block";
    }

    if (emailScanner) {
        emailScanner.style.display = "none";
    }
}


function showEmailScanner() {
    const websiteScanner = document.getElementById("website-scanner");
    const emailScanner = document.getElementById("email-scanner");

    if (websiteScanner) {
        websiteScanner.style.display = "none";
    }

    if (emailScanner) {
        emailScanner.style.display = "block";
    }
}


/* =========================================================
   RISK METER
========================================================= */

function updateRiskMeter() {

    const marker = document.querySelector(".risk-meter-marker");

    if (!marker) {
        console.log("Risk meter marker not found.");
        return;
    }

    let risk = parseFloat(marker.getAttribute("data-risk"));

    if (isNaN(risk)) {
        risk = 0;
    }

    // Keep score between 0 and 100
    risk = Math.max(0, Math.min(100, risk));

    // Move marker
    marker.style.left = risk + "%";

    console.log("Risk score:", risk);
    console.log("Marker position:", marker.style.left);
}


/* =========================================================
   PAGE LOAD
========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    updateRiskMeter();

});


/* =========================================================
   EXTRA SAFETY
   Handles cases where page loads dynamically
========================================================= */

window.addEventListener("load", function () {

    updateRiskMeter();

});