import datetime
import spacy
import pdfplumber
import pandas as pd
import re
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

class ResumeAnalyzer:
    def __init__(self, skills_csv_path: str = None):
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError(
                "SpaCy model 'en_core_web_lg' not found. Install via: python -m spacy download en_core_web_lg"
            )
        self.known_skills = self._load_skills(skills_csv_path) if skills_csv_path else []
        self.current_year = datetime.datetime.now().year
        self.skill_blacklist = {'http', 'github', 'contribution', 'january', 'february', 'company', 'owner'}

    def _load_skills(self, csv_path: str) -> List[str]:
        try:
            df = pd.read_csv(csv_path)
            skills = list(df.stack().dropna().astype(str).str.lower().unique())
            return sorted(skills, key=lambda x: (-len(x), x))
        except Exception as e:
            raise RuntimeError(f"Failed to load skills CSV: {e}")

    def extract_text(self, resume_path: str) -> str:
        try:
            with pdfplumber.open(resume_path) as pdf:
                return self._clean_text("\n".join(
                    page.extract_text() or '' for page in pdf.pages
                ))
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")

    def analyze(self, resume_text: str, job_data: Dict) -> Dict:
        resume_text = self._clean_text(resume_text)
        
        cv_data = {
            'personal_info': self._extract_personal_info(resume_text),
            'skills': self._extract_skills(resume_text),
            'experience': self._format_experience(self._extract_experience(resume_text)),
            'education': self._extract_education(resume_text),
            'links': self._extract_links(resume_text)
        }

        job_desc = self._clean_text(f"{job_data.get('description', '')} {job_data.get('requirements', '')}")
        required_skills = self._extract_skills(job_desc)
        
        similarity_score = self._calculate_semantic_similarity(resume_text, job_desc)
        skill_score = self._calculate_skill_score(cv_data['skills'], required_skills)
        experience_score = self._calculate_experience_score(resume_text, job_data.get('position', ''))
        education_score = self._calculate_education_score(resume_text, job_desc)

        compatibility_score = self._calculate_compatibility_score(
            similarity_score,
            skill_score,
            experience_score,
            education_score
        )

        return {
            'compatibility_score': compatibility_score,
            'missing_skills': self._find_missing_skills(cv_data['skills'], required_skills),
            'experience_match': experience_score >= 0.8,
            'education_match': education_score >= 0.8,
            'cv_data': cv_data['personal_info'],
            'score_breakdown': self._format_score_breakdown(
                similarity_score,
                skill_score,
                experience_score,
                education_score
            )
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[•●–—]', '-', text)
        return text.strip()

    def _extract_personal_info(self, text: str) -> Dict:
        doc = self.nlp(text)
        name = next((ent.text for ent in doc.ents if ent.label_ == 'PERSON'), '')
        email = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', text)
        phone = re.search(r'\b\+?[\d\s()-]{10,}\d\b', text)
        
        return {
            'name': str(name),
            'email': str(email.group()) if email else '',
            'phone': str(re.sub(r'\D', '', phone.group())) if phone else '',
            'links': [str(link) for link in self._extract_links(text)]
        }

    def _extract_skills(self, text: str) -> List[str]:
        section_text = self._extract_section(text, r'skills|technical skills|competencies')
        skills = []
        
        # Process skills section
        for line in re.split(r'\n|,|;', section_text):
            line = re.sub(r'[\d\.%]|years?|http[s]?://\S+', '', line, flags=re.I).strip()
            if 3 <= len(line) <= 40:
                skills.extend(self._process_skill_line(line))
        
        # Fallback to noun phrases
        if not skills:
            return self._extract_noun_phrases(text)
        
        # Final cleaning and deduplication
        return sorted(set(
            skill.lower().strip()
            for skill in skills
            if self._is_valid_skill(skill)
        ), key=lambda x: (-len(x), x))

    def _process_skill_line(self, line: str) -> List[str]:
        skills = []
        for skill in re.split(r'\s{2,}|- |/|•', line):
            skill = skill.strip()
            if 3 <= len(skill) <= 40:
                matched = self._match_known_skill(skill)
                if matched:
                    skills.append(matched)
        return skills

    def _is_valid_skill(self, skill: str) -> bool:
        return bool(
            re.search(r'[a-z]', skill) and
            not any(blacklisted in skill.lower() for blacklisted in self.skill_blacklist) and
            not re.search(r'@|\.com|\d|year|month|http', skill)
        )

    def _match_known_skill(self, skill: str) -> Optional[str]:
        if not self.known_skills:
            return skill
        
        result = process.extractOne(
            skill, self.known_skills,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85
        )
        return result[0] if result else None

    def _extract_section(self, text: str, section_name: str) -> str:
        section_pattern = re.compile(
            rf'^\s*({section_name})\s*:*\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        lines = []
        capture = False
        
        for line in text.split('\n'):
            if section_pattern.match(line.strip()):
                capture = True
                continue
            if capture:
                if re.match(r'^\s*$', line) or re.match(r'^\s*[A-Z][A-Z ]+:', line):
                    break
                lines.append(line.strip())
        
        return ' '.join(lines)

    def _extract_noun_phrases(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if 3 < len(chunk.text) < 50 
            and not any(token.is_stop for token in chunk)
            and not chunk.text.isnumeric()
        ]

    def _extract_experience(self, text: str) -> List[Dict]:
        experiences = []
        # Extract explicit duration mentions
        for match in re.finditer(r'(\d+)\+?\s*years?', text, re.I):
            experiences.append({'type': 'duration', 'value': int(match.group(1))})
        
        # Extract date ranges
        for match in re.finditer(r'(\d{4})\s*[-—]\s*(\d{4}|present)', text, re.I):
            start = int(match.group(1))
            end = self.current_year if match.group(2).lower() == 'present' else int(match.group(2))
            experiences.append({'type': 'range', 'years': end - start})
        
        return experiences

    def _format_experience(self, experiences: List[Dict]) -> List[str]:
        formatted = []
        total_years = 0
        
        for exp in experiences:
            if exp['type'] == 'duration':
                total_years += exp['value']
                formatted.append(f"{exp['value']}+ years experience")
            elif exp['type'] == 'range':
                total_years += exp.get('years', 0)
                formatted.append(f"{exp.get('years', 0)} years experience")
        
        if total_years > 0:
            formatted.insert(0, f"Total: {total_years} years")
            
        return formatted

    def _calculate_compatibility_score(self, similarity: float, skills: float, exp: float, edu: float) -> float:
        weighted = (0.4 * similarity) + (0.4 * skills) + (0.15 * exp) + (0.05 * edu)
        return float(np.round(weighted * 100, 2))

    def _format_score_breakdown(self, similarity: float, skills: float, exp: float, edu: float) -> Dict:
        return {
            'semantic_similarity': float(np.round(similarity * 100, 2)),
            'skill_coverage': float(np.round(skills * 100, 2)),
            'experience_match': float(np.round(exp * 100, 2)),
            'education_match': float(np.round(edu * 100, 2))
        }

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        if not doc1.vector_norm or not doc2.vector_norm:
            return 0.0
            
        similarity = cosine_similarity(
            doc1.vector.reshape(1, -1),
            doc2.vector.reshape(1, -1)
        )[0][0]
        
        return float(np.clip(similarity, 0.0, 1.0))

    def _calculate_skill_score(self, cv_skills: List[str], required_skills: List[str]) -> float:
        if not required_skills:
            return 1.0
            
        matched = sum(1 for skill in required_skills if skill.lower() in cv_skills)
        return float(matched / len(required_skills))

    def _calculate_experience_score(self, text: str, position: str) -> float:
        experiences = self._extract_experience(text)
        total = sum(
            exp['value'] if exp['type'] == 'duration' else exp.get('years', 0)
            for exp in experiences
        )
        
        position = position.lower()
        required = 5 if 'senior' in position else 2 if 'junior' in position else 3
        return float(min(total / required, 1.0)) if required > 0 else 0.0

    def _extract_education(self, text: str) -> List[str]:
        education = []
        degree_pattern = r'\b(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|PhD)\b'
        for match in re.finditer(fr'({degree_pattern}.*?)(?=\n\w+|\Z)', text, re.I):
            edu = match.group(0).strip()
            if len(edu) > 5 and not re.search(r'http|@', edu):
                education.append(edu)
        return education

    def _calculate_education_score(self, resume_text: str, job_desc: str) -> float:
        required_degrees = {'bachelor', 'master', 'phd'}
        candidate_degrees = {
            'bachelor' if re.search(r'\b(b\.?s|bachelor)\b', resume_text, re.I) else None,
            'master' if re.search(r'\b(m\.?s|master)\b', resume_text, re.I) else None,
            'phd' if re.search(r'\b(phd|doctorate)\b', resume_text, re.I) else None
        }
        candidate_degrees.discard(None)
        
        if not required_degrees:
            return 1.0
        return float(len(candidate_degrees & required_degrees) / len(required_degrees))

    def _extract_links(self, text: str) -> List[str]:
        return list(set(re.findall(r'https?://[^\s/$.?#].[^\s]*', text)))

    def _find_missing_skills(self, cv_skills: List[str], required_skills: List[str]) -> List[str]:
        return sorted({
            skill.lower() 
            for skill in required_skills 
            if skill.lower() not in cv_skills
            and len(skill) > 3
        })