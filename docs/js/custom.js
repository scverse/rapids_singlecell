document.addEventListener("DOMContentLoaded", function() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        let logoElement = document.querySelector("[alt='Logo']");
        logoElement.src = logoElement.src.replace('logo_light.svg', 'logo_dark.svg');
    }
});
