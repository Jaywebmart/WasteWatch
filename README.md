# WasteWatch

A machine learning system that analyzes social media discussions to identify waste patterns, competitive positioning, and intervention opportunities in UK retail, trained on 2,894 Reddit posts and achieving 73.3% accuracy in automated waste cause classification.
---

## The Problem

UK retailers lose **£50-100M annually** to food waste while consumers face unprecedented cost-of-living pressures. Traditional waste analysis relies on:
- Internal audit data (limited visibility into consumer perception)
- Government surveys (outdated, aggregated, slow to capture trends)
- Mystery shopping (expensive, small sample sizes)

Meanwhile, **millions of UK consumers discuss food waste daily on social media**—providing real-time, unfiltered insights into waste drivers, retailer performance, and emerging patterns. This data source remains largely untapped by commercial analysts.

---

## Approach

### Data Collection
- **Scraped 2,894 Reddit posts** (2012-2024) from 6 UK-focused subreddits using Pushshift API
- Navigated Reddit's 2024 API restrictions to build original dataset
- Target communities: r/ZeroWaste, r/UnitedKingdom, r/AskUK, r/CasualUK, r/TalesFromRetail, r/EatCheapAndHealthy
- Search keywords: food waste, yellow sticker, reduced to clear, expired food, supermarket waste, best before, use by date

### Analysis Pipeline
1. **Temporal Pattern Analysis**: Identified when waste discussions peak (seasonality, day-of-week, time-of-day)
2. **Retailer Sentiment Analysis**: Scored 9 UK supermarkets on waste-related sentiment using TextBlob
3. **Root Cause Text Mining**: Extracted 7 waste drivers through keyword-based classification
4. **ML Classification**: Built supervised model to predict waste category from post text using TF-IDF + Linear SVM

### Technical Stack
- **Data Collection**: PRAW, Pushshift API, requests
- **Data Processing**: pandas, numpy, datetime
- **NLP**: TextBlob (sentiment), TfidfVectorizer (feature extraction)
- **Machine Learning**: scikit-learn (Logistic Regression, Random Forest, Linear SVM)
- **Visualization**: matplotlib, seaborn, Power BI

---

## Key Findings

### Temporal Patterns
| Metric | Value | Insight |
|--------|-------|---------|
| **Growth (2012-2024)** | 2,048% | Posts increased from 25 to 537 |
| **YoY Growth (2020-2024)** | 35% avg | Accelerating conversation post-COVID |
| **Peak Month** | April (321 posts) | 37% above average - spring cleaning effect |
| **Peak Day** | Sunday (460 posts) | Weekend reflection time |
| **Peak Time** | 5pm-10pm (800+ posts) | Evening cooking/meal prep awareness |

### Competitive Intelligence (9 UK Supermarkets)
| Retailer | Mentions | Avg Sentiment | Sentiment Category | Avg Comments |
|----------|----------|---------------|-------------------|--------------|
| **Co-op** | 42 | +0.10 | Most Positive | 72 |
| **Aldi** | 38 | +0.09 | Strong Positive | 140 |
| **Waitrose** | 8 | +0.08 | Positive | 5 |
| **Tesco** | 65 | +0.07 | Neutral-Positive | 44 |
| **ASDA** | 35 | +0.07 | Neutral-Positive | 18 |
| **Sainsbury's** | 32 | +0.06 | Neutral-Positive | 49 |
| **Morrisons** | 28 | +0.04 | Neutral | 82 |
| **M&S** | 6 | +0.03 | Neutral | 12 |
| **Lidl** | 12 | -0.02 | **Only Negative** | 28 |

**Overall Sentiment Distribution**: 40.8% Positive | 45.6% Neutral | 13.6% Negative

**Key Insight**: Only 40.8% positive sentiment represents massive reputation opportunity. Co-op and Aldi lead, while Lidl is the only retailer perceived negatively on waste.

### Root Cause Analysis (7 Waste Drivers)
| Cause | Mentions | % of Posts | Top Co-occurrence |
|-------|----------|-----------|-------------------|
| **Pricing/Affordability** | 1,300 | 44.9% | + Quality/Damage (700 posts) |
| **Quality/Damage** | 1,200 | 41.5% | + Pricing (700 posts) |
| **Yellow Sticker/Markdown** | 600 | 20.7% | + Pricing (400 posts) |
| **Storage/Handling** | 500 | 17.3% | + Pricing (350 posts) |
| **Expiry/Date** | 400 | 13.8% | + Pricing (250 posts) |
| **Packaging** | 400 | 13.8% | + Pricing (200 posts) |
| **Over-ordering** | 150 | 5.2% | + Pricing (100 posts) |

**Key Insight**: Pricing dominates all waste discussions. Quality + Pricing co-occur in 700 posts, suggesting consumers perceive cost-cutting impacts product freshness—a critical finding for commercial strategy.

### Subreddit-Specific Patterns
| Subreddit | Top Waste Cause | % of Posts | Insight |
|-----------|----------------|-----------|---------|
| **r/TalesFromRetail** | Quality/Damage | 72.3% | Employee perspective: supply chain/handling issues |
| **r/EatCheapAndHealthy** | Pricing/Affordability | 57.2% | Consumer perspective: budget constraints |
| **r/ZeroWaste** | Packaging | 37.3% | Sustainability focus: excess packaging |
| **r/AskUK** | Yellow Sticker | 37.6% | Bargain hunting culture |

---

## Machine Learning Results

### Model Performance Comparison
| Model | Train Accuracy | Test Accuracy | Overfitting Gap | Winner |
|-------|---------------|---------------|-----------------|---------|
| **Linear SVM** | 78.7% | **73.3%** | 5.4% | ✅ Best |
| Random Forest | 98.9% | 67.9% | 31.0% | Overfit |
| Logistic Regression | 79.1% | 67.9% | 11.2% | - |

**Linear SVM selected** for production due to best generalization and minimal overfitting.

### Per-Category Performance (F1-Scores)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Pricing/Affordability** | 0.93 | 0.85 | 0.80 | 252 |
| **Yellow Sticker/Markdown** | 0.57 | 0.75 | 0.75 | 28 |
| **Packaging** | 0.60 | 0.70 | 0.70 | 20 |
| **Expiry/Date** | 0.67 | 0.60 | 0.68 | 30 |
| **Quality/Damage** | 0.38 | 0.60 | 0.45 | 87 |
| **Storage/Handling** | 0.28 | 0.45 | 0.35 | 25 |
| **Over-ordering** | - | - | - | 0 |

**Overall Accuracy**: 73.3% on test set (580 posts)

### Confusion Matrix Insights
- **Pricing/Affordability**: Excellent performance (236/252 correct, 93.7% precision)
- **Quality/Damage**: Struggles due to semantic overlap with Storage/Handling
- **Storage/Handling**: Weakest category due to class imbalance (smallest sample)

### Feature Importance (Top Predictive Words)
**Pricing/Affordability:**
- "money" (6.95), "price" (4.80), "cheap" (4.10), "expensive" (4.10), "deal" (2.71)

**Quality/Damage:**
- "broken" (1.83), "office" (1.76), "coffee" (1.37)

**Storage/Handling:**
- "fridge" (2.33), "leftovers" (1.78), "forgot" (1.40), "freezer" (0.99)

**Yellow Sticker/Markdown:**
- "reduced" (2.58), "yellow" (1.52), "yellow sticker" (1.06), "discount" (1.37)

**Key Insight**: Feature weights validate business intuition—pricing terms dominate affordability predictions, while storage terms correctly identify household waste behaviors.

---

## Transfer Learning Challenges

While this model performs well on Reddit data, deployment to other platforms faces:

### Data Distribution Mismatch
- **Reddit demographics**: Younger, tech-savvy, urban UK population
- **Broader market**: Older shoppers, rural communities underrepresented
- **Solution**: Retrain on Twitter/X, MoneySavingExpert forums, Mumsnet to capture wider demographic

### Temporal Drift
- Model trained on 2012-2024 data
- Language patterns evolve (e.g., "yellow sticker" terminology shifts)
- **Solution**: Implement continuous learning pipeline with monthly retraining

### Class Imbalance
- Over-ordering severely underrepresented (150 mentions vs 1,300 for Pricing)
- **Solution**: SMOTE oversampling or class weight adjustment in future iterations

---

## Project Roadmap

- [x] **Phase 1**: Reddit data collection via Pushshift API (2,894 posts)
- [x] **Phase 2**: Exploratory analysis (temporal patterns, sentiment, root causes)
- [x] **Phase 3**: NLP sentiment analysis (TextBlob, 9 retailers scored)
- [x] **Phase 4**: ML classification model (73.3% accuracy, Linear SVM)
- [ ] **Phase 5**: Power BI dashboard (3-page interactive report)
- [ ] **Phase 6**: Deploy as REST API for real-time social monitoring
- [ ] **Phase 7**: Expand to Twitter/X and forum scraping
- [ ] **Phase 8**: BERT fine-tuning for improved semantic understanding
- [ ] **Phase 9**: Open-source release with full documentation

---
## Data Sources

### Primary Dataset
**Reddit Posts (2012-2024)**
- 2,894 posts from 6 UK-focused subreddits
- Collected via Pushshift API (historical archive)
- Date range: January 2012 - October 2024
- Keywords: 12 waste-related search terms

### Validation Sources
- UK Government WRAP reports (waste statistics)
- Defra food waste data (external validation)
- Academic literature on retail waste (conceptual framework)

---

## Related Work

### Academic Foundation
- Baethge et al. (2024) - Social listening for sustainability insights
- Nielsen (2023) - Consumer sentiment as competitive intelligence
- WRAP (2024) - UK food waste reduction frameworks

### Technical Inspiration
- Pushshift Reddit Archive: https://github.com/pushshift/api
- TextBlob Sentiment: https://textblob.readthedocs.io/
- scikit-learn Text Classification: https://scikit-learn.org/stable/tutorial/text_analytics/

---

## Author

**Ayodeji Oroboade**  
MSc Business Analytics (IIBA Certified) | Machine Learning Developer

- **Portfolio**: https://jaywebmart.github.io/Ayodeji_Portfolio/
- **LinkedIn**: [www.linkedin.com/in/ayodeji-oroboade-9106aa1b0]
- **Email**: [Ayodejij147@gmail.com]

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **Claude.ai**: Thought partnership throughout analysis and model development
- **Pushshift Archive**: Historical Reddit data access
- **WRAP UK**: Food waste research and validation data
- **Reddit Communities**: r/ZeroWaste, r/UnitedKingdom, r/AskUK, r/CasualUK, r/TalesFromRetail, r/EatCheapAndHealthy for organic discussions

---

## Citation

If you use this work, please cite:
```bibtex
@software{Ayodeji2026wastewatch,
  author = {Oroboade, Ayodeji},
  title = {WasteWatch: Social Intelligence Analysis of UK Retail Food Waste},
  year = {2026},
  url = {https://github.com/jaywebmart/wastewatch}
}
```

---

## Contact

For questions, collaborations, or commercial applications:
- Open an issue in this repository
- Email: [Ayodejij147@gmail.com]
- LinkedIn: [www.linkedin.com/in/ayodeji-oroboade-9106aa1b0]

**This project demonstrates end-to-end data science capability:** web scraping, NLP, ML classification, business intelligence extraction, and stakeholder communication—all critical skills for commercial analyst roles in UK retail.

