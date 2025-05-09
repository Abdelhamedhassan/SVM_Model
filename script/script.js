
 const toggleBtn = document.getElementById('themeToggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      document.documentElement.classList.toggle('light-theme');
      localStorage.setItem('theme', document.documentElement.classList.contains('light-theme') ? 'light' : 'dark');
    });
  }

  // Load theme preference on page load
  window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
      document.documentElement.classList.add('light-theme');
    }
   });


      const fileInput = document.getElementById('file-upload');
    const predictBtn = document.getElementById('predict-btn');

    if (fileInput && predictBtn) {
        fileInput.addEventListener('change', function () {
            const inputElem = /** @type {HTMLInputElement} */ (fileInput);
            /** @type {HTMLButtonElement} */(predictBtn).disabled = !(inputElem.files && inputElem.files.length);
        });
    }

