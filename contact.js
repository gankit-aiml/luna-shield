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
  
    // FAQ Accordion functionality
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
      const question = item.querySelector('.faq-question');
      
      question.addEventListener('click', () => {
        // Close all other items
        faqItems.forEach(otherItem => {
          if (otherItem !== item && otherItem.classList.contains('active')) {
            otherItem.classList.remove('active');
          }
        });
        
        // Toggle current item
        item.classList.toggle('active');
      });
    });
  
    // Form submission
    const contactForm = document.getElementById('contactForm');
    
    contactForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Get form values
      const formData = new FormData(contactForm);
      const formValues = Object.fromEntries(formData.entries());
      
      // Here you would typically send the data to your server
      console.log('Form submitted:', formValues);
      
      // Show success message (in a real app, you'd handle the server response)
      alert('Thank you for your message! We will get back to you soon.');
      contactForm.reset();
    });
  });