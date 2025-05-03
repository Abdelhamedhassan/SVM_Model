// Function to update the active navigation link
function updateActiveNavLink() {
   // Get the current page from URL
   let currentPage = window.location.pathname;
   
   // Map paths to nav link IDs
   if (currentPage === '/' || currentPage === '/template/index.html') {
       currentPage = 'home';
   } else if (currentPage === '/prediction' || currentPage === '/template/prediction.html') {
       currentPage = 'prediction';
   } else if (currentPage === '/dashboard' || currentPage === '/template/dashboard.html') {
       currentPage = 'dashboard';
   }
   
   // Update active state for all navigation links
   const navLinks = document.querySelectorAll('.nav-link');
   navLinks.forEach(link => {
       const linkId = link.getAttribute('data-page');
       if (linkId === currentPage) {
           link.classList.add('active');
       } else {
           link.classList.remove('active');
       }
   });
}

// Rest of your JavaScript code...