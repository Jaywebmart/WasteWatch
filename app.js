// WasteWatch JavaScript - Interactive Features

// ========== SMOOTH SCROLL ==========
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

// ========== ROOT CAUSES CHART ==========
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('causesChart');
    
    if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [
                    'Pricing/Affordability',
                    'Quality/Damage',
                    'Yellow Sticker',
                    'Storage/Handling',
                    'Expiry/Date',
                    'Packaging',
                    'Over-ordering'
                ],
                datasets: [{
                    label: 'Number of Mentions',
                    data: [1300, 1200, 600, 500, 400, 400, 150],
                    backgroundColor: [
                        '#e74c3c',
                        '#e67e22',
                        '#f39c12',
                        '#3498db',
                        '#2ecc71',
                        '#1abc9c',
                        '#9b59b6'
                    ],
                    borderWidth: 0,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Waste Causes by Frequency',
                        font: {
                            size: 16,
                            weight: '600'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.x;
                                const total = 2894; // Total posts
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${value} mentions (${percentage}% of posts)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Mentions'
                        },
                        grid: {
                            color: '#ecf0f1'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
});

// ========== ACTIVE NAV HIGHLIGHTING ==========
window.addEventListener('scroll', () => {
    let current = '';
    const sections = document.querySelectorAll('section[id]');
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });
    
    document.querySelectorAll('.nav a').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(current)) {
            link.classList.add('active');
        }
    });
});

// ========== STATS COUNTER ANIMATION ==========
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.innerHTML = value.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Trigger counter animation when stats come into view
const observerOptions = {
    threshold: 0.5,
    rootMargin: '0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statNumber = entry.target.querySelector('.stat-number');
            if (statNumber && !statNumber.classList.contains('animated')) {
                const endValue = parseInt(statNumber.textContent.replace(/,/g, ''));
                if (!isNaN(endValue)) {
                    animateValue(statNumber, 0, endValue, 2000);
                    statNumber.classList.add('animated');
                }
            }
        }
    });
}, observerOptions);

document.querySelectorAll('.stat-card').forEach(card => {
    observer.observe(card);
});

// ========== CONSOLE EASTER EGG ==========
console.log('%c🌍 WasteWatch', 'color: #2ecc71; font-size: 24px; font-weight: bold;');
console.log('%cLike what you see? The code is open source!', 'color: #3498db; font-size: 14px;');
console.log('%cGitHub: https://github.com/yourusername/wastewatch', 'color: #7f8c8d; font-size: 12px;');
console.log('%cBuilt by Ayodeji Ogunleye', 'color: #2c3e50; font-size: 12px;');
