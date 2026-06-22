"""
Scoring engine for intelligent candidate ranking.
Combines multiple feature signals into a final score with reasoning.
"""

from typing import Dict, List, Tuple, Any
import math


class CandidateScorer:
    """Score and rank candidates based on JD requirements."""
    
    def __init__(self):
        # Weight scheme optimized for the Senior AI Engineer role
        self.weights = {
            # Core technical fit
            'experience_perfect_fit': 0.15,
            'core_skill_embeddings': 0.10,
            'core_skill_vector_db': 0.10,
            'core_skill_ranking': 0.08,
            'core_skill_python': 0.08,
            'core_skill_retrieval': 0.08,
            
            # Nice-to-have technical
            'core_skill_llm_finetuning': 0.04,
            'core_skill_distributed': 0.03,
            'core_skill_nlp': 0.03,
            
            # Career signals
            'has_product_experience': 0.10,
            'ml_experience_years': 0.08,
            'production_shipping_signal': 0.08,
            'services_only_career': -0.05,  # Penalty
            
            # Engagement & Availability
            'recruiter_responsiveness': 0.05,
            'recent_activity': 0.04,
            'open_to_work': 0.03,
            'notice_period': 0.03,
            
            # Profile quality
            'title_relevance': 0.05,
            'skill_breadth': 0.04,
            'endorsement_signal': 0.03,
            'assessment_scores': 0.03,
            'profile_completeness': 0.02,
            'github_activity': 0.02,
            
            # Education & soft signals
            'education_tier': 0.02,
            'relevant_degree': 0.02,
            'verification_score': 0.02,
            'career_stability': 0.02,
            'location_fit': 0.02,
            'industry_product': 0.02,
            'company_size_signal': 0.01,
            'market_interest': 0.01,
            
            # Lower priority signals
            'interview_completion': 0.01,
            'offer_acceptance': 0.01,
        }
    
    def score_candidate(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Score a candidate based on extracted features.
        Returns (score, reasoning) tuple.
        """
        
        # Check for disqualifiers
        if features.get('disqualifier_flag', 0) > 0.7:
            reason = "Disqualified: Career profile does not match JD requirements (pure research, services-only, or insufficient production experience)."
            return 0.0, reason
        
        # Check for honeypots
        honeypot = features.get('honeypot_score', 0)
        if honeypot > 0.7:
            reason = "Flagged: Profile contains impossible data patterns (honeypot)."
            return 0.0, reason
        
        # Base score calculation
        score = 0.0
        score_components = {}
        
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                feature_value = features[feature_name]
                # Handle negative weights (penalties)
                contribution = feature_value * weight
                score += contribution
                
                # Track for reasoning
                if contribution != 0:
                    score_components[feature_name] = contribution
        
        # Normalize score to 0-1 range (accounting for negative weights)
        # Maximum possible positive score
        max_positive = sum(w for w in self.weights.values() if w > 0)
        if max_positive > 0:
            score = max(0, min(1.0, score / max_positive))
        
        # Apply honeypot penalty
        score = score * (1 - honeypot * 0.5)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            features=features,
            score=score,
            top_components=sorted(
                score_components.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]  # Top 3 contributing factors
        )
        
        return score, reasoning
    
    def _generate_reasoning(
        self,
        features: Dict[str, float],
        score: float,
        top_components: List[Tuple[str, float]]
    ) -> str:
        """
        Generate human-readable reasoning for a candidate's score.
        """
        
        # Extract key facts
        exp_years = features.get('experience_perfect_fit', 0)
        ml_exp = features.get('ml_experience_years', 0)
        product_exp = features.get('has_product_experience', 0)
        shipping = features.get('production_shipping_signal', 0)
        recruiter_resp = features.get('recruiter_responsiveness', 0)
        recent = features.get('recent_activity', 0)
        notice = features.get('notice_period', 0)
        embeddings = features.get('core_skill_embeddings', 0)
        vector_db = features.get('core_skill_vector_db', 0)
        ranking = features.get('core_skill_ranking', 0)
        retrieval = features.get('core_skill_retrieval', 0)
        github = features.get('github_activity', 0)
        title_rel = features.get('title_relevance', 0)
        assessment = features.get('assessment_scores', 0)
        
        # Build component descriptions
        components = []
        
        # Experience fit
        if exp_years >= 0.8:
            components.append("strong experience alignment (6-8 years ideal range)")
        elif exp_years >= 0.6:
            components.append("acceptable experience level")
        
        # Technical depth
        tech_skills = []
        if embeddings >= 0.8:
            tech_skills.append("embeddings/semantic search")
        if vector_db >= 0.8:
            tech_skills.append("vector databases")
        if ranking >= 0.7:
            tech_skills.append("ranking systems")
        if retrieval >= 0.7:
            tech_skills.append("retrieval architecture")
        
        if tech_skills:
            components.append(f"expertise in {', '.join(tech_skills)}")
        
        # Product/ML background
        if product_exp >= 0.8 and shipping >= 0.8:
            components.append("proven shipping track record in product companies")
        elif product_exp >= 0.8:
            components.append("product company experience")
        if ml_exp >= 0.8:
            components.append("substantial applied ML background")
        
        # Engagement signals
        if recruiter_resp >= 0.7:
            components.append(f"responsive to recruiters ({recruiter_resp:.0%})")
        if recent >= 0.9:
            components.append("recently active on platform")
        
        # Availability
        if notice >= 0.9:
            components.append("available with short notice period")
        
        # Additional signals
        if github >= 0.7:
            components.append("active GitHub contributor")
        if assessment >= 0.7:
            components.append("strong platform skill assessments")
        
        # Concerns
        concerns = []
        if recruiter_resp < 0.3 and recruiter_resp > 0:
            concerns.append("low recruiter response rate")
        if recent < 0.5:
            concerns.append("inactive on platform recently")
        if notice < 0.5:
            concerns.append("longer notice period required")
        if exp_years < 0.6:
            concerns.append("experience outside ideal range")
        
        # Construct reasoning
        if score >= 0.85:
            if len(components) >= 3:
                reason = f"Excellent fit. {'; '.join(components[:3])}. "
            else:
                reason = f"Excellent fit. {'; '.join(components)}. "
        elif score >= 0.70:
            if len(components) >= 2:
                reason = f"Strong match. {'; '.join(components[:2])}. "
            else:
                reason = f"Strong match. {'; '.join(components)}. "
        elif score >= 0.50:
            if len(components) >= 2:
                reason = f"Moderate fit. {'; '.join(components[:2])}. "
            else:
                reason = f"Moderate fit. {'; '.join(components)}. "
        else:
            reason = "Limited fit for role requirements. "
        
        # Add concerns if any
        if concerns:
            reason += f"Concerns: {', '.join(concerns)}. "
        
        # Add final note
        if 0 <= score < 0.3:
            reason += "Below recommended threshold."
        
        return reason.strip()


class RankingEngine:
    """Main ranking engine combining scoring with post-processing."""
    
    def __init__(self):
        self.scorer = CandidateScorer()
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        features_dict: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Rank all candidates and return sorted list.
        """
        
        scored_candidates = []
        
        for candidate in candidates:
            candidate_id = candidate.get('candidate_id')
            
            if candidate_id not in features_dict:
                continue
            
            features = features_dict[candidate_id]
            score, reasoning = self.scorer.score_candidate(features)
            
            scored_candidates.append({
                'candidate_id': candidate_id,
                'score': score,
                'reasoning': reasoning,
                'features': features,
            })
        
        # Sort by score (descending) then by candidate_id (for deterministic tiebreaking)
        scored_candidates.sort(
            key=lambda x: (-x['score'], x['candidate_id'])
        )
        
        # Add ranks
        for i, candidate in enumerate(scored_candidates, 1):
            candidate['rank'] = i
        
        return scored_candidates
    
    def get_top_n(
        self,
        ranked_candidates: List[Dict[str, Any]],
        n: int = 100
    ) -> List[Dict[str, Any]]:
        """Get top N candidates."""
        return ranked_candidates[:n]