from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_skills(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "SKILL"]]))

def calculate_similarity(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(score[0][0]) * 100, 2)

def match_skills(extracted_skills, required_skills):
    matched = [skill for skill in required_skills if skill.lower() in map(str.lower, extracted_skills)]
    missing = [skill for skill in required_skills if skill.lower() not in map(str.lower, extracted_skills)]
    return matched, missing

def generate_insights(experience_level, matched_skills, missing_skills):
    insights = []
    if len(matched_skills) / (len(matched_skills) + len(missing_skills)) > 0.7:
        insights.append("Strong match based on required skills.")
    else:
        insights.append("You might want to add more skills to match the job description.")

    if experience_level == 'senior':
        insights.append("Emphasize leadership and project management experience.")
    return insights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    experience_level = request.form.get('experience_level', 'mid')
    required_skills = request.form.getlist('required_skills')

    resume_text = extract_text_from_pdf(file)
    extracted_skills = extract_skills(resume_text)
    similarity_score = calculate_similarity(resume_text, job_description)

    matched_skills, missing_skills = match_skills(extracted_skills, required_skills)
    insights = generate_insights(experience_level, matched_skills, missing_skills)

    return jsonify({
        "similarity_score": similarity_score,
        "extracted_skills": extracted_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "insights": insights,
        "resume_text": resume_text
    })

if __name__ == '__main__':
    app.run(debug=True)
