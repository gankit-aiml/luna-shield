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
  
    // Initialize charts
    initializeCharts();
  });
  
  function initializeCharts() {
    // Accuracy Over Time Chart
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    new Chart(accuracyCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [
          {
            label: 'Real Videos',
            data: [91.2, 92.5, 93.1, 93.8, 94.5, 95.2, 96.2],
            borderColor: '#4361ee',
            backgroundColor: 'rgba(67, 97, 238, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
          },
          {
            label: 'Deepfakes',
            data: [85.3, 86.7, 87.9, 88.5, 89.2, 89.8, 90.7],
            borderColor: '#f72585',
            backgroundColor: 'rgba(247, 37, 133, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.raw + '%';
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            min: 80,
            max: 100,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  
    // Performance Pie Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    new Chart(performanceCtx, {
      type: 'doughnut',
      data: {
        labels: ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
        datasets: [{
          data: [45, 3, 48, 4],
          backgroundColor: [
            '#4cc9f0',
            '#f8961e',
            '#4361ee',
            '#f72585'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        cutout: '70%',
        plugins: {
          legend: {
            position: 'bottom'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.raw || 0;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = Math.round((value / total) * 100);
                return `${label}: ${percentage}% (${value})`;
              }
            }
          }
        }
      }
    });
  
    // Benchmark Comparison Chart
    const benchmarkCtx = document.getElementById('benchmarkChart').getContext('2d');
    new Chart(benchmarkCtx, {
      type: 'bar',
      data: {
        labels: ['Luna Shield', 'Industry Avg.', 'Open Source', 'Commercial A'],
        datasets: [
          {
            label: 'Accuracy',
            data: [93.4, 86.2, 82.7, 89.5],
            backgroundColor: '#4361ee',
            borderRadius: 4
          },
          {
            label: 'Speed (videos/hr)',
            data: [3200, 1800, 1200, 2500],
            backgroundColor: '#4cc9f0',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label === 'Accuracy') {
                  return label + ': ' + context.raw + '%';
                } else {
                  return label + ': ' + context.raw;
                }
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                if (this.scale.id === 'y') {
                  return value + (this.scale._ticks[0].label === 'Accuracy' ? '%' : '');
                }
                return value;
              }
            }
          }
        }
      }
    });
  }
