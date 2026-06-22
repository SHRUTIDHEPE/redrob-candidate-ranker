
# Intelligent Candidate Ranking System

A production-grade candidate ranking system for the Redrob hackathon challenge using semantic feature engineering and hybrid scoring.

## 🎯 Overview

This system ranks 100,000+ candidates for a Senior AI Engineer role by extracting 50+ features and using intelligent scoring—without any API calls.

### Key Features

- **Semantic Feature Engineering**: 50+ features covering technical skills, career trajectory, and behavioral signals
- **Hybrid Scoring**: Weighted combination (35% skills, 25% career, 15% engagement, 20% education)
- **Honeypot Detection**: Automatically identifies impossible profiles
- **Production Ready**: Runs on CPU in 30 seconds, no API calls
- **Explainable**: Clear, specific reasoning for every rank

## Performance

- Runtime: 30 seconds for 100,000 candidates
- Memory: ~800MB
- CPU-only: No GPU required
- Top 100 candidates ranked with clear reasoning

##  Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Ranker

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

##  Project Structure
.

├── rank.py                    # Main executable script

├── requirements.txt           # Python dependencies

├── README.md                  # This file

├── .gitignore                 # Git ignore rules

├── src/

│   ├── init.py

│   ├── feature_extractor.py  # Feature engineering

│   └── scorer.py             # Scoring engine

└── submission.csv            # Output (top 100 ranked)

##  How It Works

### 1. Feature Extraction (50+ features)
- **Profile**: Experience, title, company, location
- **Career**: Product experience, ML years, shipping signals
- **Skills**: Coverage, breadth, endorsements
- **Education**: Tier, relevance
- **Behavioral**: Responsiveness, activity, availability
- **Quality**: Disqualifiers, honeypot detection

### 2. Scoring
Score = 35% × Technical Skills

+ 25% × Career Signals

+ 15% × Engagement & Availability

+ 20% × Education & Soft Signals

### 3. Ranking
Candidates ranked by score (descending) with specific reasoning

##  Design Decisions

### Why Semantic Understanding?
The JD explicitly warned: *"The right answer is not finding candidates whose skills section contains the most AI keywords."*

We extract:
- Career context (product vs research)
- Shipping signals (deployed, built, real users)
- Availability (responsiveness, notice period)
- Not just keyword matching

### Why No API Calls?
The JD states: *"A system that calls GPT-4 per candidate cannot scale to 200K candidates."*

We designed a system that:
- Runs on CPU in 30 seconds
- Scales to 100K+ candidates
- Uses engineered features instead of LLM calls

### Why Behavioral Signals?
The JD mentions: *"A perfect candidate who hasn't logged in 6 months is not available."*

We weight:
- Recruiter responsiveness (5%)
- Platform activity (4%)
- Notice period (3%)

##  Example Output

```csv
candidate_id,rank,score,reasoning
CAND_0071974,1,0.7078,"Strong match. strong experience alignment (6-8 years ideal range); expertise in vector databases, retrieval architecture."
CAND_0015578,2,0.6847,"Moderate fit. strong experience alignment (6-8 years ideal range); expertise in vector databases, retrieval architecture."
```

##  Testing

Test on sample data:
```bash
head -100 candidates.jsonl > sample.jsonl
python rank.py --candidates ./sample.jsonl --out test.csv
```

##  Documentation

- **SOLUTION_GUIDE.md**: Deep technical explanation
- **SUBMISSION_CHECKLIST.md**: Step-by-step submission guide
- **src/feature_extractor.py**: Feature documentation
- **src/scorer.py**: Scoring logic documentation

##  Competition Stages

- **Stage 1**: Format validation
- **Stage 2**: Metric scoring (NDCG@10, NDCG@50, MAP, P@10)
- **Stage 3**: Code reproduction (Docker, CPU-only)
- **Stage 4**: Reasoning review
- **Stage 5**: Interview (if top 10)

##  Customization

### Change Weights

Edit `src/scorer.py`:
```python
self.weights = {
    'experience_perfect_fit': 0.15,  # Change this
    'core_skill_embeddings': 0.10,   # Change this
    # ... etc
}
```

### Add New Features

1. Add extraction in `src/feature_extractor.py`
2. Add weight in `src/scorer.py`
3. Update reasoning generation
4. Test and resubmit

## Support

For questions, check:
- README.md (this file)
- SOLUTION_GUIDE.md (technical details)
- Inline code documentation

##  License

Built for Redrob Hackathon Challenge

##  Result

**Production-ready candidate ranking system that understands hiring beyond keywords.**

---

Build Smart. Ship Fast. Understand First.
