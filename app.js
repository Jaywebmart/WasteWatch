// WasteWatch — Charts

document.addEventListener('DOMContentLoaded', function() {

    // ============================================================
    // CHART 1: Waste causes by frequency (the original chart, kept)
    // ============================================================
    const causesCanvas = document.getElementById('causesChart');
    if (causesCanvas) {
        new Chart(causesCanvas, {
            type: 'bar',
            data: {
                labels: [
                    'Pricing / Affordability',
                    'Quality / Damage',
                    'Yellow Sticker / Markdown',
                    'Storage / Handling',
                    'Expiry / Date',
                    'Packaging',
                    'Over-ordering'
                ],
                datasets: [{
                    label: 'Posts mentioning this cause',
                    data: [1259, 1130, 570, 540, 430, 360, 153],
                    backgroundColor: '#006d77',
                    borderWidth: 0,
                    barThickness: 22
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a1a',
                        titleFont: { family: "'Inter', sans-serif", size: 12, weight: '600' },
                        bodyFont: { family: "'IBM Plex Mono', monospace", size: 12 },
                        padding: 10,
                        callbacks: {
                            label: function(ctx) {
                                const v = ctx.parsed.x;
                                const pct = ((v / 2894) * 100).toFixed(1);
                                return `${v.toLocaleString()} posts (${pct}% of corpus)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: '#efede7', drawBorder: false },
                        ticks: {
                            font: { family: "'IBM Plex Mono', monospace", size: 11 },
                            color: '#555555'
                        },
                        title: {
                            display: true,
                            text: 'Number of posts',
                            font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                            color: '#555555',
                            padding: { top: 10 }
                        }
                    },
                    y: {
                        grid: { display: false, drawBorder: false },
                        ticks: {
                            font: { family: "'Inter', sans-serif", size: 12 },
                            color: '#1a1a1a'
                        }
                    }
                }
            }
        });
    }

    // ============================================================
    // CHART 2: Retailer sentiment vs engagement (bubble chart)
    // ============================================================
    const retailerCanvas = document.getElementById('retailerChart');
    if (retailerCanvas) {
        const retailers = [
            { name: 'Co-op',       sentiment: 0.102, engagement: 71.2,  mentions: 39 },
            { name: 'Aldi',        sentiment: 0.077, engagement: 140.1, mentions: 23 },
            { name: 'Tesco',       sentiment: 0.075, engagement: 44.5,  mentions: 67 },
            { name: 'Waitrose',    sentiment: 0.075, engagement: 2.0,   mentions: 4  },
            { name: 'ASDA',        sentiment: 0.072, engagement: 14.6,  mentions: 26 },
            { name: "Sainsbury's", sentiment: 0.056, engagement: 48.3,  mentions: 21 },
            { name: 'Morrisons',   sentiment: 0.040, engagement: 81.6,  mentions: 21 },
            { name: 'Lidl',        sentiment: -0.001, engagement: 30.8, mentions: 5  }
        ];

        // Scale bubble size by mention count
        const maxMentions = Math.max(...retailers.map(r => r.mentions));
        const bubbleData = retailers.map(r => ({
            x: r.sentiment,
            y: r.engagement,
            r: 6 + (r.mentions / maxMentions) * 18,
            label: r.name,
            mentions: r.mentions
        }));

        new Chart(retailerCanvas, {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'UK Retailers',
                    data: bubbleData,
                    backgroundColor: 'rgba(0, 109, 119, 0.55)',
                    borderColor: '#006d77',
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a1a',
                        titleFont: { family: "'Inter', sans-serif", size: 12, weight: '600' },
                        bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
                        padding: 10,
                        callbacks: {
                            title: function(items) { return items[0].raw.label; },
                            label: function(ctx) {
                                const d = ctx.raw;
                                return [
                                    `Sentiment: ${d.x >= 0 ? '+' : ''}${d.x.toFixed(3)}`,
                                    `Avg comments: ${d.y.toFixed(1)}`,
                                    `Mentions: ${d.mentions}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Average sentiment (−1 to +1)',
                            font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                            color: '#555555',
                            padding: { top: 10 }
                        },
                        grid: { color: '#efede7', drawBorder: false },
                        ticks: {
                            font: { family: "'IBM Plex Mono', monospace", size: 11 },
                            color: '#555555'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Average comments per post',
                            font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                            color: '#555555'
                        },
                        grid: { color: '#efede7', drawBorder: false },
                        ticks: {
                            font: { family: "'IBM Plex Mono', monospace", size: 11 },
                            color: '#555555'
                        }
                    }
                }
            }
        });

        // Add labels on the bubbles after render
        const plugin = {
            id: 'bubble-labels',
            afterDatasetsDraw(chart) {
                const ctx = chart.ctx;
                const dataset = chart.data.datasets[0];
                const meta = chart.getDatasetMeta(0);
                meta.data.forEach((point, i) => {
                    const d = dataset.data[i];
                    ctx.save();
                    ctx.font = "500 11px 'Inter', sans-serif";
                    ctx.fillStyle = '#1a1a1a';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(d.label, point.x, point.y - point.options.radius - 8);
                    ctx.restore();
                });
            }
        };
        Chart.register(plugin);
        // Re-render to apply plugin
        const retailerChart = Chart.getChart(retailerCanvas);
        if (retailerChart) retailerChart.update();
    }
});
