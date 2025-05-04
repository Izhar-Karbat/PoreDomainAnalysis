// This file is used by wkhtmltopdf with the --run-script option to ensure fonts are properly embedded
// It runs in the context of the web page before PDF rendering

function prepareFontsForPdf() {
    // Create a div to test all characters we need to render
    var fontTest = document.createElement('div');
    fontTest.style.fontFamily = "'Open Sans', Arial, sans-serif";
    fontTest.style.visibility = 'hidden';
    fontTest.style.position = 'absolute';
    fontTest.style.top = '-10000px';
    
    // Include a wide range of characters to ensure font is loaded properly
    fontTest.innerHTML = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{};:,.<>/?`~";
    
    // Add the test element to the document body
    document.body.appendChild(fontTest);
    
    // Force a reflow to ensure font loading
    void fontTest.offsetHeight;
    
    // Print debug info
    console.log("Font preparation complete");
}

// Run the function
prepareFontsForPdf();