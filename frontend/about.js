document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu functionality
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
  
    mobileMenuButton.addEventListener('click', function(e) {
      e.stopPropagation();
      mobileMenu.style.display = mobileMenu.style.display === 'flex' ? 'none' : 'flex';
    });
  
    // Close mobile menu when clicking outside
    document.addEventListener('click', function() {
      mobileMenu.style.display = 'none';
    });
  
    // Prevent mobile menu from closing when clicking inside it
    mobileMenu.addEventListener('click', function(e) {
      e.stopPropagation();
    });
  
    // Highlight active link in mobile menu
    const currentPage = window.location.pathname.split('/').pop();
    const mobileLinks = document.querySelectorAll('.mobile-menu-link');
    
    mobileLinks.forEach(link => {
      if (link.getAttribute('href') === currentPage) {
        link.classList.add('active');
      }
    });
  });
