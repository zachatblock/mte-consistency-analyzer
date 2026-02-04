// W3A Test Analysis - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add active class to current nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // Form validation helpers
    window.validateForm = function(formId) {
        const form = document.getElementById(formId);
        if (!form) return false;
        
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                field.classList.add('error');
                isValid = false;
            } else {
                field.classList.remove('error');
            }
        });
        
        return isValid;
    };

    // Progress bar animation
    window.updateProgress = function(percentage, elementId = 'progress-bar') {
        const progressBar = document.getElementById(elementId);
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.textContent = Math.round(percentage) + '%';
        }
    };

    // Show/hide loading spinner
    window.showLoading = function(show = true) {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
    };

    // Display alerts
    window.showAlert = function(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container');
        if (!alertContainer) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        
        alertContainer.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    };

    // File upload handling
    window.handleFileUpload = function(inputId, callback) {
        const input = document.getElementById(inputId);
        if (!input) return;
        
        input.addEventListener('change', function(e) {
            const files = e.target.files;
            if (files.length > 0 && callback) {
                callback(files);
            }
        });
    };
});
