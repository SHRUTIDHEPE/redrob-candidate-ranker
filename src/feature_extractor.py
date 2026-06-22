"""
Feature extraction for intelligent candidate ranking.
Extracts semantic and structured features without API calls.
"""

import json
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import Counter
import math


class FeatureExtractor:
    """Extract features from JD and candidate profiles."""
    
    # Core skills extracted from JD requirements
    CORE_SKILLS = {
        'embeddings': ['embedding', 'sentence-transformer', 'bgE', 'e5', 'openai embed'],
        'vector_db': ['vector', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'opensearch', 'elasticsearch'],
        'ranking': ['ranking', 'ranker', 'ndcg', 'mrr', 'map', 'learning-to-rank'],
        'retrieval': ['retrieval', 'retrieval-augmented', 'rag', 'dense retrieval', 'hybrid search', 'bm25'],
        'llm_finetuning': ['fine-tun', 'lora', 'qlora', 'peft', 'instruction tuning'],
        'python': ['python'],
        'ml': ['machine learning', 'ml engineer'],
        'nlp': ['nlp', 'natural language processing'],
        'distributed': ['distributed', 'spark', 'kafka', 'async'],
    }
    
    # Disqualifiers mentioned in JD
    DISQUALIFIERS = {
        'pure_research': ['research', 'academic', 'phd only'],
        'no_production': ['no production', 'never shipped', 'research only'],
        'only_recent_llm': ['chatgpt', 'langchain', 'openai only'],
        'no_code_18m': ['architect', 'tech lead', 'no code'],
        'services_only': ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'],
    }
    
    # JD Requirements (extracted from job description)
    JD_REQUIREMENTS = {
        'min_experience': 5,
        'max_experience': 9,
        'preferred_experience': 6,  # 6-8 is ideal
        'required_areas': ['embeddings', 'vector_db', 'ranking', 'python'],
        'nice_to_have': ['llm_finetuning', 'distributed', 'nlp'],
        'disqualifying_profiles': [
            'pure_research',
            'only_recent_llm',
            'no_code_18m',
            'services_only',
        ],
        'preferred_locations': ['Pune', 'Noida', 'Delhi', 'Bangalore', 'Hyderabad'],
    }
    
    def __init__(self):
        self.jd_requirements = self.JD_REQUIREMENTS
        
    def extract_candidate_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features for a candidate.
        Returns a feature vector as dictionary.
        """
        features = {}
        
        # Profile-level features
        features.update(self._extract_profile_features(candidate.get('profile', {})))
        
        # Career history features
        features.update(self._extract_career_features(
            candidate.get('career_history', []),
            candidate.get('profile', {})
        ))
        
        # Skills features
        features.update(self._extract_skills_features(candidate.get('skills', [])))
        
        # Education features
        features.update(self._extract_education_features(candidate.get('education', [])))
        
        # Behavioral signals
        features.update(self._extract_behavioral_features(candidate.get('redrob_signals', {})))
        
        # Disqualifier check
        features['disqualifier_flag'] = self._check_disqualifiers(candidate)
        
        # Honeypot detection
        features['honeypot_score'] = self._detect_honeypot(candidate)
        
        return features
    
    def _extract_profile_features(self, profile: Dict) -> Dict[str, float]:
        """Extract features from basic profile."""
        features = {}
        
        # Experience level match with JD (ideal 6-8 years)
        exp = profile.get('years_of_experience', 0)
        if 6 <= exp <= 8:
            features['experience_perfect_fit'] = 1.0
        elif 5 <= exp <= 9:
            features['experience_perfect_fit'] = 0.8
        elif 4 <= exp <= 10:
            features['experience_perfect_fit'] = 0.6
        else:
            features['experience_perfect_fit'] = max(0, 1.0 - abs(exp - 6.5) / 10)
        
        # Current title relevance
        title = profile.get('current_title', '').lower()
        title_keywords = ['engineer', 'ml', 'ai', 'data', 'senior', 'principal', 'architect']
        title_match = sum(1 for kw in title_keywords if kw in title) / len(title_keywords)
        features['title_relevance'] = title_match
        
        # Company size signal (prefer growth-stage / larger companies)
        company_size = profile.get('current_company_size', '')
        size_ranking = {
            '1-10': 0.3, '11-50': 0.5, '51-200': 0.6,
            '201-500': 0.7, '501-1000': 0.8, '1001-5000': 0.85,
            '5001-10000': 0.9, '10001+': 0.95
        }
        features['company_size_signal'] = size_ranking.get(company_size, 0.5)
        
        # Industry relevance (product vs services)
        industry = profile.get('current_industry', '').lower()
        if any(x in industry for x in ['software', 'tech', 'ai', 'saas', 'fintech', 'product']):
            features['industry_product'] = 1.0
        elif any(x in industry for x in ['services', 'consulting', 'staffing']):
            features['industry_product'] = 0.0
        else:
            features['industry_product'] = 0.5
        
        # Location flexibility (India preferred)
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        if 'india' in country or 'in' == country:
            features['location_fit'] = 1.0
        elif any(city.lower() in location for city in self.jd_requirements['preferred_locations']):
            features['location_fit'] = 0.95
        elif any(x in location for x in ['pune', 'noida', 'delhi', 'bangalore']):
            features['location_fit'] = 0.9
        else:
            features['location_fit'] = 0.5
        
        return features
    
    def _extract_career_features(self, career_history: List[Dict], profile: Dict) -> Dict[str, float]:
        """Extract features from career history."""
        features = {}
        
        if not career_history:
            return {
                'has_product_experience': 0.0,
                'production_shipping_signal': 0.0,
                'career_stability': 0.0,
                'ml_experience_years': 0.0,
            }
        
        # Check for product company experience
        product_keywords = ['software', 'tech', 'ai', 'saas', 'fintech', 'product', 'startup']
        services_keywords = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
        
        product_roles = []
        services_only = True
        total_ml_years = 0
        
        for role in career_history:
            company = role.get('company', '').lower()
            industry = role.get('industry', '').lower()
            title = role.get('title', '').lower()
            duration = role.get('duration_months', 0) / 12
            
            # Check if services company
            if not any(x in company for x in services_keywords):
                services_only = False
            
            # Check for product roles
            if any(kw in company or kw in industry for kw in product_keywords):
                product_roles.append(role)
            
            # Track ML years
            if any(x in title for x in ['ml', 'ai', 'data engineer', 'ml engineer']):
                total_ml_years += duration
        
        features['has_product_experience'] = 1.0 if product_roles else 0.0
        features['services_only_career'] = 0.0 if services_only else 1.0
        
        # ML experience year match (JD wants 4-5 years applied ML)
        if 4 <= total_ml_years <= 6:
            features['ml_experience_years'] = 1.0
        elif 3 <= total_ml_years <= 7:
            features['ml_experience_years'] = 0.8
        elif 2 <= total_ml_years <= 8:
            features['ml_experience_years'] = 0.6
        else:
            features['ml_experience_years'] = max(0, 1.0 - abs(total_ml_years - 4.5) / 5)
        
        # Career stability (prefer longer stints, not job hopper)
        if len(career_history) > 0:
            avg_tenure = sum(r.get('duration_months', 0) for r in career_history) / len(career_history) / 12
            if avg_tenure >= 2:
                features['career_stability'] = 1.0
            elif avg_tenure >= 1.5:
                features['career_stability'] = 0.8
            else:
                features['career_stability'] = 0.5
        else:
            features['career_stability'] = 0.5
        
        # Shipping signal (look for "shipped", "production", "real users" in descriptions)
        shipped_indicators = ['shipped', 'production', 'deployed', 'built', 'implemented', 'real users', 'end-to-end']
        shipping_count = 0
        for role in career_history:
            desc = role.get('description', '').lower()
            if any(x in desc for x in shipped_indicators):
                shipping_count += 1
        features['production_shipping_signal'] = min(1.0, shipping_count / max(1, len(career_history)))
        
        return features
    
    def _extract_skills_features(self, skills: List[Dict]) -> Dict[str, float]:
        """Extract features from skills section."""
        features = {}
        
        skill_names = {s.get('name', '').lower() for s in skills}
        endorsement_sum = sum(s.get('endorsements', 0) for s in skills)
        
        # Core skills match
        for skill_group, keywords in self.CORE_SKILLS.items():
            match_count = sum(
                1 for keyword in keywords
                if any(keyword.lower() in sname for sname in skill_names)
            )
            features[f'core_skill_{skill_group}'] = min(1.0, match_count / 2)
        
        # Overall skill breadth
        features['skill_breadth'] = min(1.0, len(skills) / 15)
        
        # Skill endorsement signal
        if endorsement_sum > 100:
            features['endorsement_signal'] = 1.0
        elif endorsement_sum > 50:
            features['endorsement_signal'] = 0.8
        elif endorsement_sum > 20:
            features['endorsement_signal'] = 0.6
        else:
            features['endorsement_signal'] = min(1.0, endorsement_sum / 20)
        
        # High-proficiency skills
        advanced_skills = sum(1 for s in skills if s.get('proficiency') in ['advanced', 'expert'])
        features['advanced_skills_count'] = min(1.0, advanced_skills / 5)
        
        return features
    
    def _extract_education_features(self, education: List[Dict]) -> Dict[str, float]:
        """Extract features from education."""
        features = {}
        
        if not education:
            features['education_tier'] = 0.3
            features['relevant_degree'] = 0.0
            return features
        
        # Education tier
        tier_scores = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.5, 'tier_4': 0.2, 'unknown': 0.3}
        avg_tier = sum(tier_scores.get(e.get('tier', 'unknown'), 0.3) for e in education) / len(education)
        features['education_tier'] = avg_tier
        
        # Relevant degree
        relevant_keywords = ['computer science', 'engineering', 'mathematics', 'physics', 'statistics', 'ai', 'ml']
        relevant_count = sum(
            1 for e in education
            if any(kw in e.get('field_of_study', '').lower() for kw in relevant_keywords)
        )
        features['relevant_degree'] = 1.0 if relevant_count > 0 else 0.0
        
        return features
    
    def _extract_behavioral_features(self, signals: Dict) -> Dict[str, float]:
        """Extract behavioral engagement signals."""
        features = {}
        
        # Availability signals
        features['open_to_work'] = 1.0 if signals.get('open_to_work_flag', False) else 0.5
        
        # Notice period (JD wants <30 days)
        notice = signals.get('notice_period_days', 60)
        if notice <= 30:
            features['notice_period'] = 1.0
        elif notice <= 60:
            features['notice_period'] = 0.7
        elif notice <= 90:
            features['notice_period'] = 0.5
        else:
            features['notice_period'] = 0.2
        
        # Recruiter engagement
        response_rate = signals.get('recruiter_response_rate', 0.0)
        features['recruiter_responsiveness'] = response_rate
        
        # Platform activity recency
        last_active = signals.get('last_active_date', '2020-01-01')
        if isinstance(last_active, str):
            try:
                days_since_active = (datetime.strptime('2024-12-31', '%Y-%m-%d') - 
                                   datetime.strptime(last_active, '%Y-%m-%d')).days
            except:
                days_since_active = 180
        else:
            days_since_active = 180
        
        if days_since_active <= 30:
            features['recent_activity'] = 1.0
        elif days_since_active <= 90:
            features['recent_activity'] = 0.9
        elif days_since_active <= 180:
            features['recent_activity'] = 0.7
        else:
            features['recent_activity'] = max(0, 1.0 - (days_since_active - 180) / 365)
        
        # Recruiter interest
        search_appearances = signals.get('search_appearance_30d', 0)
        saved_by_recruiters = signals.get('saved_by_recruiters_30d', 0)
        features['market_interest'] = min(1.0, (search_appearances + saved_by_recruiters * 2) / 20)
        
        # Profile quality
        completeness = signals.get('profile_completeness_score', 50) / 100
        features['profile_completeness'] = completeness
        
        # GitHub signal
        github_score = signals.get('github_activity_score', -1)
        if github_score >= 0:
            features['github_activity'] = github_score / 100
        else:
            features['github_activity'] = 0.0
        
        # Skill assessments
        skill_assessments = signals.get('skill_assessment_scores', {})
        if skill_assessments:
            avg_assessment = sum(skill_assessments.values()) / len(skill_assessments) / 100
            features['assessment_scores'] = min(1.0, avg_assessment)
        else:
            features['assessment_scores'] = 0.0
        
        # Interview/offer history
        interview_rate = signals.get('interview_completion_rate', 0.0)
        offer_rate = signals.get('offer_acceptance_rate', -1)
        
        features['interview_completion'] = interview_rate
        if offer_rate >= 0:
            features['offer_acceptance'] = offer_rate
        else:
            features['offer_acceptance'] = 0.0
        
        # Verification signals
        verified_signals = sum([
            signals.get('verified_email', False),
            signals.get('verified_phone', False),
            signals.get('linkedin_connected', False)
        ]) / 3
        features['verification_score'] = verified_signals
        
        return features
    
    def _check_disqualifiers(self, candidate: Dict) -> float:
        """Check for major disqualifiers. Returns 1.0 if disqualified, 0.0 if not."""
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        summary = profile.get('summary', '').lower()
        
        disqualifier_score = 0.0
        
        # Pure research only
        research_keywords = ['research', 'academic', 'phd', 'lab', 'paper', 'published']
        if sum(summary.count(kw) for kw in research_keywords) >= 3:
            if 'production' not in summary and 'shipped' not in summary:
                disqualifier_score = max(disqualifier_score, 0.5)
        
        # Only recent LLM experience
        all_text = summary + ' '.join(r.get('description', '') for r in career_history).lower()
        if ('langchain' in all_text or 'chatgpt' in all_text or 'openai' in all_text) and 'embedding' not in all_text:
            if not any(x in all_text for x in ['2022', '2021', '2020', '2019']):  # pre-LLM era experience
                disqualifier_score = max(disqualifier_score, 0.3)
        
        # Services company only
        company_names = [r.get('company', '').lower() for r in career_history]
        services_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
        if company_names and all(any(s in c for s in services_companies) for c in company_names if c):
            disqualifier_score = max(disqualifier_score, 0.8)
        
        return disqualifier_score
    
    def _detect_honeypot(self, candidate: Dict) -> float:
        """
        Detect impossible profiles (honeypots).
        Returns score 0-1 where 1.0 = definitely honeypot.
        """
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        
        honeypot_score = 0.0
        
        # Check 1: More years of experience than possible
        exp_years = profile.get('years_of_experience', 0)
        if exp_years > 40:
            honeypot_score = max(honeypot_score, 0.9)
        
        # Check 2: Skills with impossible proficiency/duration
        for skill in skills:
            prof = skill.get('proficiency', 'intermediate')
            duration = skill.get('duration_months', 0)
            endorsements = skill.get('endorsements', 0)
            
            # Expert with <6 months experience is suspicious
            if prof == 'expert' and duration < 6:
                honeypot_score = max(honeypot_score, 0.6)
            
            # Too many endorsements (>200) for a junior skill
            if duration < 12 and endorsements > 100:
                honeypot_score = max(honeypot_score, 0.5)
        
        # Check 3: Company tenure vs company age
        for role in career_history:
            end_date = role.get('end_date')
            start_date = role.get('start_date')
            company = role.get('company', '')
            
            try:
                if start_date and end_date:
                    start = datetime.strptime(start_date, '%Y-%m-%d')
                    end = datetime.strptime(end_date, '%Y-%m-%d')
                    if (end - start).days < 0:  # end before start
                        honeypot_score = max(honeypot_score, 0.9)
            except:
                pass
        
        # Check 4: Unrealistic skill count
        if len(skills) > 30:
            honeypot_score = max(honeypot_score, 0.4)
        
        return honeypot_score


def extract_jd_features() -> Dict[str, Any]:
    """Extract features directly from JD requirements."""
    return {
        'required_skills': ['embeddings', 'vector_db', 'ranking', 'python'],
        'nice_to_have': ['llm_finetuning', 'distributed', 'nlp'],
        'experience_ideal_years': (6, 8),
        'experience_acceptable_years': (5, 9),
    }