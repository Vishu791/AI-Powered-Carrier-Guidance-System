import sys
import os
import json
import joblib
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import tempfile

from sentence_transformers import SentenceTransformer, util

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QListWidget,
    QScrollArea, QStackedWidget, QMessageBox, QFrame, QTabWidget,
    QSizePolicy, QGridLayout, QLineEdit, QDialog, QDialogButtonBox,
    QFormLayout, QGroupBox, QGraphicsOpacityEffect, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QSize, QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtGui import QFont, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

plt.style.use("dark_background")


# ---------------- RESPONSIVE GLASS UI STYLE ----------------
APP_STYLE = """
* {
    font-family: 'Segoe UI', 'Inter', sans-serif;
    color: #eee;
}

QMainWindow { 
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a3a3a, stop:1 #1f1f1f);
}

QWidget#rootContainer,
QWidget#contentPage {
    background: transparent;
}

/* Glass & card surfaces */
QFrame#glassFrame {
    background: rgba(38, 41, 48, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
}

QFrame#glassFrame:hover {
    border-color: rgba(255, 255, 255, 0.18);
    background: rgba(50, 53, 60, 0.92);
}

QFrame#inputFrame {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(60, 60, 60, 0.95), stop:1 rgba(34, 34, 34, 0.95));
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 28px;
    padding: 8px;
}

QFrame#resultCard {
    background: rgba(35, 35, 35, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 22px;
    transition: all 120ms ease-out;
}

QFrame#resultCard:hover {
    border-color: rgba(255, 255, 255, 0.25);
    background: rgba(45, 45, 45, 0.97);
}

QFrame#pathsFrame, QFrame#roadmapFrame, QFrame#specialtyFrame {
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(32, 32, 32, 0.95);
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5f5f5f, stop:1 #3d3d3d);
    color: #f8f8f8;
    border: none;
    border-radius: 16px;
    padding: 14px 22px;
    font-size: 14px;
    font-weight: 600;
    min-height: 28px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6d6d6d, stop:1 #4a4a4a);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #444, stop:1 #2e2e2e);
}

QPushButton#secondary {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

QPushButton#secondary:hover {
    background: rgba(255, 255, 255, 0.12);
}

/* Inputs */
QComboBox, QLineEdit, QTextEdit, QListWidget {
    background: rgba(20, 20, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 12px;
    font-size: 14px;
    color: #f5f5f5;
    min-height: 24px;
}

QComboBox::drop-down {
    border: none;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    width: 26px;
}

QComboBox QAbstractItemView {
    background: rgba(32, 32, 32, 0.97);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    selection-background-color: rgba(255, 255, 255, 0.12);
    font-size: 13px;
}

/* Labels */
QLabel#title {
    font-size: clamp(26px, 4vw, 34px);
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #f5f5f5;
}

QLabel#sectionTitle {
    font-size: clamp(14px, 2vw, 17px);
    font-weight: 600;
    color: #d1d5db;
}

QLabel#cardTitle {
    font-size: clamp(16px, 2.5vw, 21px);
    font-weight: 700;
    color: #f5f5f5;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    background: rgba(26, 26, 26, 0.95);
    padding: 6px;
}

QTabBar::tab {
    background: rgba(20, 20, 20, 0.85);
    color: #d4d4d4;
    padding: 12px 24px;
    margin: 6px;
    border-radius: 14px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
    min-width: 120px;
}

QTabBar::tab:selected {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.25);
    color: white;
}

QTabBar::tab:hover {
    border-color: rgba(255, 255, 255, 0.18);
}

/* Scroll areas */
QScrollArea {
    border: none;
    background: transparent;
}

/* Scrollbars */
QScrollBar:vertical {
    background: rgba(40, 40, 40, 0.8);
    width: 11px;
    border-radius: 6px;
    margin: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 6px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.4);
}
"""

def safe_load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


# ------------ CAREER DATA ------------
DOCTOR_SPECIALTIES = {
    "Cardiologist": "Heart and blood vessel specialist. Treats heart attacks, hypertension, and cardiac diseases.",
    "Neurologist": "Deals with brain and nervous system disorders like epilepsy, stroke, migraine, etc.",
    "Dermatologist": "Skin, hair and nail specialist. Handles acne, allergies, skin infections and cosmetic dermatology.",
    "Pediatrician": "Child specialist doctor, focusing on infants and children's growth, health and vaccination.",
}

CAREER_MAPPINGS = {
    "streams": ["Science", "Commerce", "Arts", "Other"],
    "fields": {
        "Science": ["Medical & Healthcare", "Engineering & Technology", "Research & Development", "Data Science & Analytics"],
        "Commerce": [
            "Finance & Accounting Path",
            "Business & Management",
            "Economics & Data",
            "Banking & Government Services",
            "Law & Corporate Governance",
            "Creative + Business Fusion",
            "Tech + Commerce",
            "International Career"
        ],
        "Arts": ["Law & Legal Services", "Psychology & Counseling", "Media & Journalism", "Design & Creative Arts"],
        "Other": ["Government Services", "Defense", "Sports", "Other"]
    },
    "roles": {
        "Medical & Healthcare": ["Doctor", "Dentist", "Nurse", "Pharmacist", "Medical Researcher"],
        "Engineering & Technology": ["Software Engineer", "Data Scientist", "Mechanical Engineer", "Civil Engineer", "Electronics Engineer", "Aerospace Engineer"],
        "Research & Development": ["Research Scientist", "Biotechnologist", "Lab Technician", "Biomedical Scientist"],
        "Data Science & Analytics": ["Data Scientist", "Business Analyst", "AI Engineer", "Data Analyst"],
        "Finance & Accounting Path": [
            "Chartered Accountant (CA)",
            "Cost & Management Accountant (CMA)",
            "Company Secretary (CS)",
            "CPA / ACCA Professional"
        ],
        "Business & Management": [
            "BBA Graduate (Marketing/HR/Finance/IB)",
            "MBA Leadership Roles",
            "Entrepreneur / Startup Founder",
            "Supply Chain & Logistics Manager",
            "Hospitality / Hotel Management",
            "Business Manager / Corporate Strategist"
        ],
        "Economics & Data": [
            "BA/BSc Economics Specialist",
            "Actuarial Scientist",
            "Business Analytics Professional",
            "Data Analyst (Business Intelligence)"
        ],
        "Banking & Government Services": [
            "Banking Officer (IBPS/SBI PO)",
            "UPSC / SSC / Railways / Defence Accounts",
            "RBI / SEBI / Finance Officer"
        ],
        "Law & Corporate Governance": [
            "B.Com + LLB Graduate",
            "Corporate Lawyer",
            "Company Secretary (CS)"
        ],
        "Creative + Business Fusion": [
            "Advertising / Digital Marketing Manager",
            "Media Management Professional",
            "Event Management Specialist",
            "Fashion Business & Retail Strategist"
        ],
        "Tech + Commerce": [
            "FinTech Product Specialist",
            "E-Commerce Manager",
            "Business IT (BCA with Specialization)",
            "Cyber Finance Compliance Analyst"
        ],
        "International Career": [
            "International Business Manager",
            "Import & Export Consultant",
            "Foreign Trade Specialist (IIFT etc.)",
            "Global CFO / Finance Professional (CPA/ACCA)"
        ],
        "Law & Legal Services": ["Lawyer", "Corporate Lawyer", "Judge", "Legal Consultant"],
        "Psychology & Counseling": ["Psychologist", "Clinical Psychologist", "Therapist", "Counselor"],
        "Media & Journalism": ["Journalist", "Content Creator", "News Anchor", "Media Strategist"],
        "Design & Creative Arts": ["Graphic Designer", "UI/UX Designer", "Animator", "Game Designer"],
        "Government Services": ["IAS Officer", "IPS Officer", "Government Clerk", "Policy Analyst"],
        "Defense": ["Army Officer", "Navy Officer", "Air Force Officer", "Defense Scientist"],
        "Sports": ["Athlete", "Sports Coach", "Fitness Trainer", "Sports Scientist"],
        "Other": ["Entrepreneur", "Consultant", "Freelancer", "Teacher"]
    }
}

# Map fields to streams for quick lookup
FIELD_TO_STREAM = {}
for stream_name, fields in CAREER_MAPPINGS["fields"].items():
    for field_name in fields:
        FIELD_TO_STREAM[field_name] = stream_name

# Build a reverse lookup from role/career to the streams where it appears
STREAM_ROLE_MAP = {}
for stream_name, fields in CAREER_MAPPINGS["fields"].items():
    for field_name in fields:
        for role_name in CAREER_MAPPINGS["roles"].get(field_name, []):
            STREAM_ROLE_MAP.setdefault(role_name, set()).add(stream_name)

# Build a reverse lookup from role/career to fields for prioritization
FIELD_ROLE_MAP = {}
for field_name, roles in CAREER_MAPPINGS["roles"].items():
    for role_name in roles:
        FIELD_ROLE_MAP.setdefault(role_name, set()).add(field_name)

ENHANCED_CAREER_DETAILS = {
    "Software Engineer": {
        "description": "Design, develop, and maintain software systems and applications. Work across various domains like web development, mobile apps, AI systems, and enterprise software.",
        "education": ["Bachelor's in Computer Science", "Bachelor's in Software Engineering", "Master's in Computer Science"],
        "skills": ["Programming", "Algorithms", "Data Structures", "Software Design", "Testing", "Debugging"],
        "salary": "₹6-25 LPA (Fresh Graduate: ₹6-12 LPA, Senior: ₹15-25 LPA+)",
        "market": "High demand with 20%+ growth expected. Opportunities in IT services, product companies, startups.",
        "pros": ["High salary potential (₹15-25 LPA+ for experienced)", "Remote work flexibility", "Global job opportunities", "Creative problem-solving", "Fast career growth", "Startup equity potential"],
        "cons": ["Long working hours during project deadlines", "Need to constantly learn new technologies", "High competition for top companies", "Can be mentally exhausting", "Age bias in some companies"],
        "roadmap": [
            "Complete 12th with PCM/CS",
            "Clear JEE/State CET for B.Tech CSE or pursue BCA",
            "Learn programming languages (Python/Java/C++/JavaScript)",
            "Build projects and contribute to GitHub",
            "Intern at IT companies during college",
            "Prepare for technical interviews (DSA, system design)",
            "Start as software developer and progress to senior roles"
        ]
    },
    "Data Scientist": {
        "description": "Extract insights from complex data using statistical analysis, machine learning, and data visualization techniques.",
        "education": ["Bachelor's in Computer Science/Statistics", "Master's in Data Science", "MBA in Analytics"],
        "skills": ["Python/R", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
        "salary": "₹8-30 LPA (Fresh: ₹8-15 LPA, Senior: ₹20-30 LPA+)",
        "market": "Rapidly growing field with high demand across industries like finance, healthcare, e-commerce.",
        "pros": ["High demand in AI/ML companies", "Excellent compensation (₹20-30 LPA+ for experienced)", "Work with cutting-edge AI technology", "Make data-driven business decisions", "Diverse industry applications", "Research opportunities"],
        "cons": ["Requires strong mathematical and statistical background", "Dealing with messy/incomplete data", "Complex problem-solving under pressure", "Need to stay updated with ML frameworks", "High expectations from stakeholders"],
        "roadmap": [
            "Complete 12th with PCM/CS",
            "Pursue B.Tech/B.Sc in CS/Statistics/Mathematics",
            "Learn Python, SQL, and statistics fundamentals",
            "Master ML frameworks (TensorFlow, scikit-learn, PyTorch)",
            "Build data science projects and Kaggle competitions",
            "Pursue internships and ML certifications",
            "Start as data analyst and progress to data scientist"
        ]
    },
    "Doctor": {
        "description": "Medical professional diagnosing and treating illnesses, injuries, and providing healthcare services to patients.",
        "education": ["MBBS (5.5 years)", "MD/MS for specialization", "Residency Training"],
        "skills": ["Medical Knowledge", "Diagnosis", "Patient Care", "Communication", "Emergency Handling"],
        "salary": "₹10-50 LPA (Junior: ₹10-15 LPA, Specialist: ₹25-50 LPA+)",
        "market": "Always in demand with stable career prospects. Opportunities in hospitals, clinics, research.",
        "pros": ["Excellent job security and demand", "High respect and prestige in society", "Opportunity to save lives and help people", "Diverse specializations (Cardiology, Neurology, etc.)", "High earning potential (₹25-50 LPA+ for specialists)", "Can start own clinic or practice"],
        "cons": ["Very long education period (5.5 years MBBS + 3 years MD/MS)", "Extremely high stress and pressure", "Long working hours (often 12+ hours)", "Emotional challenges dealing with patient suffering", "High competition for NEET seats", "Expensive medical education"],
        "roadmap": [
            "Class 11-12 with PCB (Physics, Chemistry, Biology)",
            "Crack NEET-UG to secure MBBS seat",
            "Complete MBBS (5.5 years including internship)",
            "Clear NEET-PG or INI CET for specialization",
            "Complete MD/MS + residency for roles such as Neurosurgeon, Cardiologist, etc."
        ],
        "sub_specialty_steps": {
            "Neurosurgeon": [
                "MBBS + NEET-PG",
                "MS in General Surgery",
                "MCh / DNB Super Specialization in Neurosurgery"
            ],
            "Nurse": [
                "Physics-Chemistry-Biology in 12th",
                "Qualify entrance for B.Sc. Nursing / GNM",
                "Clear state nursing council exams + internship"
            ]
        }
    },
    "Pharmacist": {
        "description": "Dispense medicines, counsel patients on correct usage, manage drug inventory, and ensure regulatory compliance in retail or clinical settings.",
        "education": ["Diploma in Pharmacy (D.Pharm)", "Bachelor of Pharmacy (B.Pharm)", "Master of Pharmacy (M.Pharm)", "Pharm.D"],
        "skills": ["Pharmacology", "Drug Dispensing", "Patient Counseling", "Inventory Management", "Regulatory Compliance", "Attention to Detail"],
        "salary": "₹3-12 LPA (Retail: ₹3-6 LPA, Hospital: ₹4-8 LPA, Industry/Senior Roles: ₹8-12 LPA)",
        "market": "Steady demand across hospitals, retail chains, government health centers, and pharmaceutical companies.",
        "pros": ["Multiple work settings (retail, hospital, industry, government)", "Option to start own medical store/pharmacy", "High trust and respect in healthcare", "Growing healthcare sector with steady demand", "Lower entry barrier than MBBS", "Flexible career paths"],
        "cons": ["Long hours standing on feet in retail", "Strict regulatory oversight and compliance", "Need to constantly stay updated with new drugs", "Moderate salary in retail (₹3-6 LPA)", "High competition for government jobs", "Initial investment for own shop"],
        "paths": [
            {
                "title": "Retail / Community Pharmacist",
                "description": "Open your own medical store or manage a chain pharmacy. Handle prescriptions, inventory, and patient guidance."
            },
            {
                "title": "Hospital / Clinical Pharmacist",
                "description": "Work with doctors inside hospitals to prepare and dispense medications, monitor drug interactions, and support patient recovery."
            },
            {
                "title": "Pharmaceutical Industry Specialist",
                "description": "Join pharma manufacturing, quality assurance, medical coding, or drug safety teams with opportunities for rapid growth."
            },
            {
                "title": "Government & Regulatory Services",
                "description": "Clear Drug Inspector or government health department exams to oversee compliance, licensing, and public drug programs."
            }
        ],
        "roadmap": [
            "Study PCB (or PCM) in Class 11-12",
            "Appear for state CET/entrance for D.Pharm or B.Pharm",
            "Complete internship + register with State Pharmacy Council",
            "Optionally pursue M.Pharm/Pharm.D for advanced roles",
            "For own shop: obtain drug license + GST + setup approvals",
            "For government roles: clear Drug Inspector/Pharmacist exams"
        ]
    },
    "Business Manager": {
        "description": "Oversee business operations, manage teams, and drive organizational growth and strategy.",
        "education": ["Bachelor's in Business Administration", "MBA", "Industry-specific certifications"],
        "skills": ["Leadership", "Strategic Planning", "Team Management", "Communication", "Problem-solving"],
        "salary": "₹8-40 LPA (Junior: ₹8-15 LPA, Senior: ₹25-40 LPA+)",
        "market": "Consistent demand across all industries. Essential for organizational success.",
        "pros": ["Leadership and decision-making authority", "Excellent compensation (₹25-40 LPA+ for senior roles)", "Diverse industry exposure", "Fast career progression to C-suite", "Networking opportunities", "Strategic impact on business"],
        "cons": ["High responsibility and accountability", "Stressful decision-making under pressure", "Work-life balance challenges", "Need to manage difficult stakeholders", "Performance pressure from top management"],
        "roadmap": [
            "Complete 12th in Commerce/Science/Arts",
            "Pursue BBA or relevant bachelor's degree",
            "Gain work experience (2-3 years recommended)",
            "Pursue MBA from reputed institute (CAT/XAT/GMAT)",
            "Start in entry-level management roles",
            "Progress to senior management positions",
            "Optionally pursue executive MBA or certifications"
        ]
    },
    "Dentist": {
        "description": "Diagnose and treat dental issues, perform oral surgeries, and provide preventive dental care to patients.",
        "education": ["BDS (Bachelor of Dental Surgery)", "MDS for specialization", "Dental Council Registration"],
        "skills": ["Dental Procedures", "Oral Surgery", "Patient Care", "Manual Dexterity", "Diagnosis"],
        "salary": "₹6-25 LPA (Fresh: ₹6-10 LPA, Specialist: ₹15-25 LPA+)",
        "market": "Steady demand in private clinics, hospitals, and government dental facilities.",
        "pros": ["Flexible work hours compared to doctors", "Can start own dental clinic", "Good income potential (₹15-25 LPA+ for specialists)", "Helping people with oral health", "Less competition than MBBS", "Shorter education than MBBS"],
        "cons": ["Long education (5 years BDS + 3 years MDS for specialization)", "Physical strain from standing and working with hands", "High equipment costs for own practice (₹10-50 lakhs)", "Need to manage clinic operations if self-employed", "Competition in urban areas"],
        "roadmap": [
            "Complete 12th with PCB",
            "Clear NEET-UG for BDS admission",
            "Complete BDS (5 years including internship)",
            "Register with Dental Council of India",
            "Optionally pursue MDS for specialization (Orthodontics, Oral Surgery, etc.)",
            "Start practice or join dental clinic/hospital"
        ]
    },
    "Nurse": {
        "description": "Provide patient care, assist doctors, administer medications, and monitor patient health in hospitals and clinics.",
        "education": ["B.Sc Nursing", "GNM (General Nursing & Midwifery)", "Post Basic B.Sc Nursing"],
        "skills": ["Patient Care", "Medical Procedures", "Communication", "Empathy", "Emergency Response"],
        "salary": "₹3-12 LPA (Staff Nurse: ₹3-6 LPA, Senior Nurse: ₹8-12 LPA)",
        "market": "High demand in hospitals, clinics, nursing homes, and community health centers.",
        "pros": ["Job security", "Opportunity to help people", "Diverse work settings", "Career progression"],
        "cons": ["Physically demanding", "Shift work", "Emotional stress", "Long hours"],
        "roadmap": [
            "Complete 12th with PCB",
            "Qualify for B.Sc Nursing/GNM entrance exams",
            "Complete nursing degree (4 years for B.Sc, 3.5 years for GNM)",
            "Register with State Nursing Council",
            "Clear nursing license exam",
            "Start as staff nurse and progress to senior roles"
        ]
    },
    "Mechanical Engineer": {
        "description": "Design, develop, and maintain mechanical systems, machinery, and manufacturing processes across industries.",
        "education": ["B.Tech in Mechanical Engineering", "M.Tech for specialization", "Industry certifications"],
        "skills": ["CAD/CAM", "Machine Design", "Thermodynamics", "Manufacturing Processes", "Project Management"],
        "salary": "₹5-20 LPA (Fresh: ₹5-8 LPA, Senior: ₹12-20 LPA+)",
        "market": "Stable demand in manufacturing, automotive, energy, and infrastructure sectors.",
        "pros": ["Diverse industry options", "Hands-on work", "Good job stability", "Technical challenges"],
        "cons": ["Can be physically demanding", "Manufacturing sector fluctuations", "Need continuous learning"],
        "roadmap": [
            "Complete 12th with PCM",
            "Clear JEE/State CET for B.Tech Mechanical",
            "Complete B.Tech (4 years)",
            "Gain internship experience",
            "Optionally pursue M.Tech in specialized areas",
            "Start as design engineer or production engineer"
        ]
    },
    "Civil Engineer": {
        "description": "Design, construct, and maintain infrastructure projects like buildings, roads, bridges, and water systems.",
        "education": ["B.Tech in Civil Engineering", "M.Tech in Structural/Transportation Engineering"],
        "skills": ["Structural Design", "Construction Management", "Surveying", "AutoCAD", "Project Planning"],
        "salary": "₹4-18 LPA (Fresh: ₹4-7 LPA, Senior: ₹10-18 LPA+)",
        "market": "Consistent demand due to infrastructure development and urbanization projects.",
        "pros": ["Tangible results", "Job stability", "Government opportunities", "Field work"],
        "cons": ["Site-based work", "Weather dependent", "Safety risks", "Long hours on sites"],
        "roadmap": [
            "Complete 12th with PCM",
            "Clear JEE/State CET for B.Tech Civil",
            "Complete B.Tech (4 years)",
            "Gain site experience through internships",
            "Optionally pursue M.Tech or get licensed",
            "Start as site engineer or design engineer"
        ]
    },
    "Electronics Engineer": {
        "description": "Design and develop electronic circuits, embedded systems, and communication devices for various applications.",
        "education": ["B.Tech in Electronics/ECE", "M.Tech in VLSI/Embedded Systems"],
        "skills": ["Circuit Design", "Embedded Systems", "Microcontrollers", "Signal Processing", "PCB Design"],
        "salary": "₹5-22 LPA (Fresh: ₹5-9 LPA, Senior: ₹12-22 LPA+)",
        "market": "Growing demand in consumer electronics, IoT, automotive, and telecommunications.",
        "pros": ["Innovation opportunities", "Diverse applications", "Good salary growth", "Tech-focused"],
        "cons": ["Rapid technology changes", "Need constant skill updates", "Complex problem-solving"],
        "roadmap": [
            "Complete 12th with PCM",
            "Clear JEE/State CET for B.Tech ECE",
            "Complete B.Tech (4 years)",
            "Learn embedded systems and microcontrollers",
            "Gain internship in electronics companies",
            "Start as design engineer or embedded systems engineer"
        ]
    },
    "Aerospace Engineer": {
        "description": "Design aircraft, spacecraft, satellites, and related systems for aviation and space industries.",
        "education": ["B.Tech in Aerospace/Aeronautical Engineering", "M.Tech for specialization"],
        "skills": ["Aerodynamics", "Aircraft Design", "Propulsion Systems", "CAD", "Simulation"],
        "salary": "₹8-30 LPA (Fresh: ₹8-12 LPA, Senior: ₹18-30 LPA+)",
        "market": "High demand in ISRO, DRDO, HAL, and private aerospace companies.",
        "pros": ["Cutting-edge technology", "Prestigious field", "Government opportunities", "Innovation"],
        "cons": ["Limited job openings", "High competition", "Requires advanced education"],
        "roadmap": [
            "Complete 12th with PCM",
            "Clear JEE for B.Tech Aerospace/Aeronautical",
            "Complete B.Tech (4 years)",
            "Pursue M.Tech for specialization",
            "Apply to ISRO, DRDO, HAL, or private aerospace firms",
            "Start as design engineer or research engineer"
        ]
    },
    "Chartered Accountant (CA)": {
        "description": "Manage financial records, conduct audits, provide tax consultancy, and ensure regulatory compliance for businesses.",
        "education": ["CA Foundation", "CA Intermediate", "CA Final", "B.Com/M.Com"],
        "skills": ["Accounting", "Auditing", "Taxation", "Financial Reporting", "GST", "Tally"],
        "salary": "₹8-30 LPA (Article: ₹2-4 LPA, Qualified CA: ₹8-15 LPA, Senior: ₹20-30 LPA+)",
        "market": "Evergreen profession with demand across all industries and businesses.",
        "pros": ["Highly respected professional certification", "Diverse opportunities (audit, tax, finance, corporate)", "Excellent earning potential (₹20-30 LPA+ for experienced)", "Can start own CA firm", "Evergreen profession with high demand", "Prestigious qualification"],
        "cons": ["Very difficult exams (low pass rates)", "Long study period (3-5 years with articleship)", "Continuous learning required (tax laws, GST changes)", "Can be repetitive work in audit", "High stress during tax season", "Long working hours in CA firms"],
        "roadmap": [
            "Complete 12th in Commerce",
            "Register for CA Foundation",
            "Clear CA Foundation exam",
            "Complete CA Intermediate (with articleship)",
            "Clear CA Final exam",
            "Start practice or join CA firm/corporate"
        ]
    },
    "Lawyer": {
        "description": "Provide legal advice, represent clients in court, draft legal documents, and ensure compliance with laws.",
        "education": ["LLB (3 years after graduation)", "BA LLB (5 years integrated)", "LLM for specialization"],
        "skills": ["Legal Research", "Argumentation", "Drafting", "Client Counseling", "Court Procedures"],
        "salary": "₹5-50 LPA (Junior: ₹5-10 LPA, Senior: ₹20-50 LPA+, Corporate: Higher)",
        "market": "Steady demand in law firms, corporate legal departments, and government services.",
        "pros": ["Highly respected profession in society", "Intellectual challenges and analytical work", "Excellent earning potential (₹20-50 LPA+ for corporate lawyers)", "Diverse specializations (corporate, criminal, civil, IP)", "Can start own law practice", "Prestigious career path"],
        "cons": ["Very long working hours (often 12+ hours)", "Extremely high stress and pressure", "Highly competitive field", "Need continuous learning of new laws", "Irregular work schedule", "High competition for top law firms"],
        "roadmap": [
            "Complete 12th in any stream",
            "Clear CLAT/AILET for BA LLB or complete graduation for LLB",
            "Complete law degree (3-5 years)",
            "Clear bar exam and register with Bar Council",
            "Start as junior associate or join law firm",
            "Specialize in corporate, criminal, or civil law"
        ]
    },
    "Psychologist": {
        "description": "Study human behavior, provide counseling, conduct therapy sessions, and help people with mental health issues.",
        "education": ["BA/B.Sc Psychology", "MA/M.Sc Psychology", "M.Phil/Ph.D for clinical practice"],
        "skills": ["Counseling", "Assessment", "Empathy", "Communication", "Research Methods"],
        "salary": "₹4-20 LPA (Fresh: ₹4-8 LPA, Clinical Psychologist: ₹10-20 LPA+)",
        "market": "Growing awareness of mental health increases demand in hospitals, clinics, and private practice.",
        "pros": ["Helping people", "Diverse specializations", "Flexible work", "Growing field"],
        "cons": ["Emotional demands", "Requires advanced degrees for clinical practice", "Licensing requirements"],
        "roadmap": [
            "Complete 12th in any stream (Arts preferred)",
            "Pursue BA/B.Sc in Psychology",
            "Complete MA/M.Sc in Psychology",
            "For clinical practice: Complete M.Phil/Ph.D",
            "Register with Rehabilitation Council of India",
            "Start practice or join hospital/clinic"
        ]
    },
    "Journalist": {
        "description": "Research, investigate, and report news stories for print, digital, and broadcast media platforms.",
        "education": ["BA in Journalism/Mass Communication", "MA in Journalism", "Diploma in Media Studies"],
        "skills": ["Writing", "Research", "Interviewing", "Communication", "Digital Media", "Video Editing"],
        "salary": "₹3-15 LPA (Reporter: ₹3-6 LPA, Senior Journalist: ₹10-15 LPA+)",
        "market": "Evolving field with opportunities in digital media, news channels, and online platforms.",
        "pros": ["Dynamic work", "Meet diverse people", "Impactful stories", "Creative expression"],
        "cons": ["Irregular hours", "Field work risks", "Deadline pressure", "Job instability in some sectors"],
        "roadmap": [
            "Complete 12th in any stream",
            "Pursue BA/MA in Journalism or Mass Communication",
            "Gain experience through internships",
            "Build portfolio with published work",
            "Start as reporter or content writer",
            "Progress to senior journalist or editor"
        ]
    },
    "UI/UX Designer": {
        "description": "Design user-friendly interfaces and experiences for websites, apps, and digital products.",
        "education": ["B.Des", "BFA", "UI/UX Design Certifications", "Relevant Bachelor's + Design Course"],
        "skills": ["Figma", "Wireframing", "Prototyping", "User Research", "Visual Design", "Interaction Design"],
        "salary": "₹5-25 LPA (Junior: ₹5-10 LPA, Senior: ₹15-25 LPA+)",
        "market": "High demand in product companies, IT firms, startups, and design agencies.",
        "pros": ["Creative work", "High demand", "Good compensation", "Remote opportunities"],
        "cons": ["Subjective feedback", "Need to stay updated", "Tight deadlines", "Competitive field"],
        "roadmap": [
            "Complete 12th (any stream)",
            "Learn design tools (Figma, Adobe XD)",
            "Study UI/UX principles and user research",
            "Build portfolio with case studies",
            "Get certified or pursue design course",
            "Start as junior designer or intern"
        ]
    }
}

HOBBY_OPTIONS = [
    "🧠 Logic & Problem Solving (Puzzles / Maths / Debugging)",
    "🎨 Creativity & Design (Drawing, Branding, Innovation)",
    "🤝 People Interaction (Guiding, Teaching, Teamwork)",
    "📊 Business & Money (Entrepreneurship, Finance, Markets)",
    "🔬 Science & Experiments (Biology/Chemistry/Physics Labs)",
    "🧩 Technology & Computers (Coding, Hardware, AI, Cybersecurity)",
    "🏛️ Law, Policy & Governance (Debate, Ethics, Justice)",
    "🌍 Environment & Nature (Wildlife, Ecology, Sustainability)",
    "🏋️‍♂️ Sports & Physical Training (Fitness, Coaching)",
    "🎭 Media & Communication (Content, Writing, Film, PR)",
    "🚀 Innovation & Future Tech (Space, EVs, Robotics, Metaverse)",
    "💗 Helping & Community Impact (Mental Health, NGOs, Social Work)"
]
FREE_TIME_OPTIONS = ["Coding/Technical Projects", "Online Courses/Learning", "Reading/Books", "Designing/Creative Work"]
SUBJECT_OPTIONS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
    "Botany", "Zoology", "Human Anatomy", "AI/ML/Data Science",
    "Economics/Commerce", "Business Management", "Design/Arts",
    "Law", "Legal Studies", "Political Science", "Civics", "Psychology"
]

SUBJECT_FIELD_MAP = {
    "Computer Science": ["Engineering & Technology", "Data Science & Analytics"],
    "Mathematics": ["Engineering & Technology", "Data Science & Analytics"],
    "Physics": ["Engineering & Technology"],
    "Chemistry": ["Medical & Healthcare", "Research & Development"],
    "Biology": ["Medical & Healthcare", "Research & Development"],
    "Botany": ["Medical & Healthcare", "Research & Development"],
    "Zoology": ["Medical & Healthcare", "Research & Development"],
    "Human Anatomy": ["Medical & Healthcare"],
    "AI/ML/Data Science": ["Data Science & Analytics"],
    "Economics/Commerce": ["Finance & Accounting Path", "Business & Management", "Economics & Data"],
    "Business Management": ["Business & Management", "Tech + Commerce", "Creative + Business Fusion"],
    "Design/Arts": ["Design & Creative Arts"],
    "Law": ["Law & Legal Services", "Law & Corporate Governance"],
    "Legal Studies": ["Law & Legal Services", "Law & Corporate Governance"],
    "Political Science": ["Law & Legal Services", "Government Services", "Law & Corporate Governance"],
    "Civics": ["Government Services", "Law & Legal Services", "Law & Corporate Governance"],
    "Psychology": ["Psychology & Counseling", "Medical & Healthcare"]
}

INTEREST_FIELD_MAP = {
    "🧠 Logic & Problem Solving (Puzzles / Maths / Debugging)": [
        "Engineering & Technology", "Data Science & Analytics", "Economics & Data", "Tech + Commerce"
    ],
    "🎨 Creativity & Design (Drawing, Branding, Innovation)": [
        "Creative + Business Fusion", "Design & Creative Arts", "International Career"
    ],
    "🤝 People Interaction (Guiding, Teaching, Teamwork)": [
        "Business & Management", "Psychology & Counseling", "Law & Corporate Governance", "Banking & Government Services"
    ],
    "📊 Business & Money (Entrepreneurship, Finance, Markets)": [
        "Finance & Accounting Path", "Business & Management", "Economics & Data", "Tech + Commerce", "International Career"
    ],
    "🔬 Science & Experiments (Biology/Chemistry/Physics Labs)": [
        "Medical & Healthcare", "Research & Development", "Biomedical & Life Sciences"
    ],
    "🧩 Technology & Computers (Coding, Hardware, AI, Cybersecurity)": [
        "Engineering & Technology", "Data Science & Analytics", "Tech + Commerce", "Innovation & Future Tech"
    ],
    "🏛️ Law, Policy & Governance (Debate, Ethics, Justice)": [
        "Law & Legal Services", "Law & Corporate Governance", "Government Services", "Banking & Government Services"
    ],
    "🌍 Environment & Nature (Wildlife, Ecology, Sustainability)": [
        "Environmental Science", "Government Services", "International Career"
    ],
    "🏋️‍♂️ Sports & Physical Training (Fitness, Coaching)": [
        "Medical & Healthcare", "Creative + Business Fusion"
    ],
    "🎭 Media & Communication (Content, Writing, Film, PR)": [
        "Creative + Business Fusion", "Media & Journalism", "Business & Management"
    ],
    "🚀 Innovation & Future Tech (Space, EVs, Robotics, Metaverse)": [
        "Engineering & Technology", "Data Science & Analytics", "Innovation & Future Tech", "International Career"
    ],
    "💗 Helping & Community Impact (Mental Health, NGOs, Social Work)": [
        "Psychology & Counseling", "Medical & Healthcare", "Law & Corporate Governance", "Banking & Government Services"
    ]
}

SCIENCE_MEDICAL_PATHWAYS = [
    {
        "label": "MBBS (Doctor / Surgeon)",
        "duration": "5.5 years",
        "entrance": "NEET",
        "careers": ["Doctor", "Surgeon", "Cardiologist", "Neurologist", "Dermatologist", "General Physician"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "BDS (Dentistry)",
        "duration": "5 years",
        "entrance": "NEET",
        "careers": ["Dentist", "Orthodontist", "Dental Surgeon", "Prosthodontist"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "BAMS / Ayurveda Medicine",
        "duration": "5.5 years",
        "entrance": "NEET / State Exams",
        "careers": ["Ayurvedic Doctor", "Panchakarma Specialist", "Ayurvedic Researcher"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "BHMS / Homeopathy",
        "duration": "5.5 years",
        "entrance": "NEET",
        "careers": ["Homeopathic Physician", "Holistic Health Consultant"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "Veterinary Science (BVSc)",
        "duration": "5.5 years",
        "entrance": "NEET / AIPVT",
        "careers": ["Veterinary Doctor", "Wildlife Vet", "Animal Nutritionist"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "BPT (Physiotherapy)",
        "duration": "4.5 years",
        "entrance": "CUET / Institute Exams",
        "careers": ["Physiotherapist", "Sports Rehab Specialist", "Occupational Therapist"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "Pharmacy (B.Pharm / PharmD)",
        "duration": "4 years (B.Pharm) / 6 years (PharmD)",
        "entrance": "CUET / GPAT / State CET",
        "careers": ["Pharmacist", "Clinical Pharmacologist", "Drug Research Scientist"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "Nursing (BSc Nursing / GNM)",
        "duration": "4 years",
        "entrance": "AIIMS / State Exams",
        "careers": ["Nurse Practitioner", "Critical Care Expert", "Nurse Educator"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "Allied Medical Sciences",
        "duration": "3-4 years",
        "entrance": "CUET / State / Institute Exams",
        "careers": ["Radiology Specialist", "Medical Lab Technologist", "Anesthesia Technologist", "OT Technologist", "Optometrist", "Audiologist"],
        "field_tags": ["Medical & Healthcare"]
    },
    {
        "label": "Biomedical & Life Sciences (Research Route)",
        "duration": "3-5 years",
        "entrance": "CUET / IISER / Private Universities",
        "careers": ["Biotechnologist", "Genetic Engineer", "Lab Scientist", "Pharma R&D Specialist"],
        "field_tags": ["Medical & Healthcare", "Research & Development"]
    },
    {
        "label": "Psychology & Mental Health",
        "duration": "3 years + PG",
        "entrance": "CUET / Institute Exams",
        "careers": ["Psychologist", "Clinical Psychologist", "Forensic Psychologist", "Therapist"],
        "field_tags": ["Psychology & Counseling", "Medical & Healthcare"]
    }
]

SCIENCE_NON_MED_PATHWAYS = [
    {
        "label": "Engineering (JEE / CET Route)",
        "duration": "4 years",
        "entrance": "JEE Main + Adv / State CET / BITSAT / VITEEE",
        "careers": ["Software Engineer", "Mechanical Engineer", "Civil Engineer", "Chemical Engineer", "Aerospace Engineer", "Automobile Engineer", "Robotics Engineer", "Marine Engineer", "Petroleum Engineer"],
        "field_tags": ["Engineering & Technology", "Data Science & Analytics"]
    },
    {
        "label": "Architecture (B.Arch)",
        "duration": "5 years",
        "entrance": "NATA / JEE Paper 2",
        "careers": ["Architect", "Urban Planner", "Interior Designer"],
        "field_tags": ["Design & Creative Arts", "Engineering & Technology"]
    },
    {
        "label": "Computer & Tech (Non-Engineering)",
        "duration": "3 years",
        "entrance": "CUET / Institute Exams",
        "careers": ["Software Developer", "Cybersecurity Analyst", "Game Developer", "UI/UX Designer"],
        "field_tags": ["Engineering & Technology", "Tech + Commerce"]
    },
    {
        "label": "Pure Sciences & Research",
        "duration": "3-5 years",
        "entrance": "CUET / IISER / ISI / TIFR",
        "careers": ["Research Scientist", "Astrophysicist", "Oceanographer", "Nanotechnologist", "ISRO/DRDO Scientist"],
        "field_tags": ["Research & Development", "Data Science & Analytics"]
    },
    {
        "label": "Aviation & Space Careers",
        "duration": "2-4 years",
        "entrance": "DGCA / NDA / IGRUA",
        "careers": ["Commercial Pilot", "Aerospace Engineer", "ATC Officer", "Space Scientist"],
        "field_tags": ["Engineering & Technology", "Defense"]
    },
    {
        "label": "Merchant Navy",
        "duration": "3-4 years",
        "entrance": "IMU-CET",
        "careers": ["Marine Engineer", "Nautical Officer", "Port Operations Manager"],
        "field_tags": ["Engineering & Technology", "International Career"]
    },
    {
        "label": "Defense Technical Route",
        "duration": "Varies",
        "entrance": "NDA / TES / DRDO Exams",
        "careers": ["Defense Engineer", "Technical Officer", "Weapon Systems Specialist"],
        "field_tags": ["Defense", "Government Services"]
    },
    {
        "label": "Maths + Finance Fusion",
        "duration": "3-5 years",
        "entrance": "CUET / Actuarial Papers",
        "careers": ["Actuarial Scientist", "Quantitative Analyst", "Financial Engineer"],
        "field_tags": ["Economics & Data", "Finance & Accounting Path"]
    },
    {
        "label": "Design + Innovation (Science Route)",
        "duration": "4 years",
        "entrance": "NID / UCEED / NIFT / CUCET",
        "careers": ["Product Designer", "Automobile Designer", "Industrial Designer", "VFX Artist"],
        "field_tags": ["Design & Creative Arts", "Creative + Business Fusion"]
    },
    {
        "label": "Government / Civil Services Route",
        "duration": "3+ years",
        "entrance": "UPSC / SSC / State PCS",
        "careers": ["IAS Officer", "Scientist-B (DRDO)", "Banking Officer"],
        "field_tags": ["Government Services", "Law & Corporate Governance"]
    }
]

SCIENCE_TRACK_PATHWAYS = {
    "Medical": SCIENCE_MEDICAL_PATHWAYS,
    "Non-Medical": SCIENCE_NON_MED_PATHWAYS
}

SCIENCE_PATH_LOOKUP = {}
SCIENCE_PATH_LABELS = {}
for focus_name, path_list in SCIENCE_TRACK_PATHWAYS.items():
    SCIENCE_PATH_LABELS[focus_name] = [path["label"] for path in path_list]
    for path in path_list:
        SCIENCE_PATH_LOOKUP[path["label"]] = path
        FIELD_TO_STREAM[path["label"]] = "Science"


def get_science_path_labels_for_focus(focus=None):
    if focus and focus in SCIENCE_TRACK_PATHWAYS:
        return SCIENCE_PATH_LABELS.get(focus, [])
    labels = []
    for value in SCIENCE_PATH_LABELS.values():
        labels.extend(value)
    return labels


def get_science_field_tags_for_focus(focus=None):
    tags = set()
    if focus and focus in SCIENCE_TRACK_PATHWAYS:
        path_list = SCIENCE_TRACK_PATHWAYS[focus]
    else:
        path_list = [path for paths in SCIENCE_TRACK_PATHWAYS.values() for path in paths]
    for path in path_list:
        tags.update(path.get("field_tags", []))
    return tags


FIELD_CAREER_CLUSTERS = {
    "Medical & Healthcare": ["Doctor", "Dentist", "Pharmacist", "Nurse", "Physiotherapist", "Ayurvedic Doctor", "Homeopathic Physician", "Veterinary Doctor", "Radiology Specialist"],
    "Engineering & Technology": ["Software Engineer", "Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Electronics Engineer", "Aerospace Engineer", "Automobile Engineer", "Robotics Engineer"],
    "Research & Development": ["Research Scientist", "Biotechnologist", "Lab Scientist", "Biomedical Scientist", "Genetic Engineer"],
    "Data Science & Analytics": ["Data Scientist", "Business Analyst", "AI Engineer", "Data Analyst", "Statistician"],
    "Environmental Science": ["Environmental Scientist", "Forestry Officer", "Marine Biologist", "Geologist"],
    "Finance & Accounting Path": ["Chartered Accountant (CA)", "Cost & Management Accountant (CMA)", "Company Secretary (CS)", "Finance / Investment Analyst"],
    "Business & Management": ["Business Manager / Corporate Strategist", "Entrepreneur / Startup Founder", "Supply Chain & Logistics Manager", "Hospitality / Hotel Management"],
    "Economics & Data": ["Economist / Policy Analyst", "Actuarial Scientist", "Business Analytics Professional", "Data Analyst"],
    "Banking & Government Services": ["Banking Officer (IBPS/SBI PO)", "UPSC / SSC / Railways / Defence Accounts", "RBI / SEBI / Finance Officer"],
    "Law & Corporate Governance": ["Lawyer", "Corporate Lawyer", "Company Secretary (CS)", "Compliance Officer"],
    "Creative + Business Fusion": ["Advertising / Digital Marketing Manager", "Media Management Professional", "Event Management Specialist", "Fashion Business & Retail Strategist", "Product Designer"],
    "Tech + Commerce": ["FinTech Product Specialist", "E-Commerce Manager", "Business IT (BCA with Specialization)", "Cyber Finance Compliance Analyst"],
    "International Career": ["International Business Manager", "Import & Export Consultant", "Foreign Trade Specialist (IIFT etc.)", "Global CFO / Finance Professional (CPA/ACCA)"],
    "Law & Legal Services": ["Lawyer", "Corporate Lawyer", "Legal Consultant", "Judge"],
    "Psychology & Counseling": ["Psychologist", "Clinical Psychologist", "Therapist", "Counselor"],
    "Media & Journalism": ["Journalist", "Content Creator", "News Anchor", "Media Strategist"],
    "Design & Creative Arts": ["Graphic Designer", "UI/UX Designer", "Animator", "Game Designer", "Interior Designer"],
    "Government Services": ["IAS Officer", "IPS Officer", "Government Clerk", "Policy Analyst"],
    "Defense": ["Army Officer", "Navy Officer", "Air Force Officer", "Defense Scientist"],
    "Sports": ["Athlete", "Sports Coach", "Fitness Trainer", "Sports Scientist"],
    "Other": ["Entrepreneur", "Consultant", "Freelancer", "Teacher"],
    "Biomedical & Life Sciences (Research Route)": ["Biotechnologist", "Genetic Engineer", "Lab Scientist", "Pharma R&D Specialist"],
    "Psychology & Mental Health": ["Psychologist", "Clinical Psychologist", "Forensic Psychologist", "Therapist"],
    "Allied Medical Sciences": ["Radiology Specialist", "Medical Lab Technologist", "Anesthesia Technologist", "OT Technologist", "Optometrist", "Audiologist"],
    "Pharmacy (B.Pharm / PharmD)": ["Pharmacist", "Clinical Pharmacologist", "Drug Research Scientist"],
    "Nursing (BSc Nursing / GNM)": ["Nurse Practitioner", "Critical Care Expert", "Nurse Educator"],
    "BPT (Physiotherapy)": ["Physiotherapist", "Sports Rehab Specialist", "Occupational Therapist"],
    "BDS (Dentistry)": ["Dentist", "Orthodontist", "Dental Surgeon"],
    "BAMS / Ayurveda Medicine": ["Ayurvedic Doctor", "Panchakarma Specialist"],
    "BHMS / Homeopathy": ["Homeopathic Physician"],
    "Veterinary Science (BVSc)": ["Veterinary Doctor", "Wildlife Vet"],
    "Engineering (JEE / CET Route)": ["Software Engineer", "Mechanical Engineer", "Civil Engineer", "Chemical Engineer", "Aerospace Engineer", "Robotics Engineer", "Marine Engineer"],
    "Architecture (B.Arch)": ["Architect", "Urban Planner", "Interior Designer"],
    "Computer & Tech (Non-Engineering)": ["Software Developer", "Cybersecurity Analyst", "Game Developer", "UI/UX Designer"],
    "Pure Sciences & Research": ["Research Scientist", "Astrophysicist", "Oceanographer", "Nanotechnologist"],
    "Aviation & Space Careers": ["Commercial Pilot", "Aerospace Engineer", "ATC Officer", "Space Scientist"],
    "Merchant Navy": ["Marine Engineer", "Nautical Officer"],
    "Defense Technical Route": ["Defense Engineer", "Technical Officer"],
    "Maths + Finance Fusion": ["Actuarial Scientist", "Quantitative Analyst", "Financial Engineer"],
    "Design + Innovation (Science Route)": ["Product Designer", "Automobile Designer", "Industrial Designer", "VFX Artist"],
    "Government / Civil Services Route": ["IAS Officer", "Scientist-B (DRDO)", "Banking Officer"],
    "Innovation & Future Tech": ["Aerospace Engineer", "Robotics Engineer", "AI Engineer", "EV Specialist"]
}

COLLEGE_INFO_BY_FIELD = {
    "Engineering & Technology": [
        {"name": "IIT Bombay", "exam": "JEE Advanced", "highlights": "Top engineering campus with world-class labs and 90%+ placements."},
        {"name": "IIT Madras", "exam": "JEE Advanced", "highlights": "Renowned for research, incubation support, and interdisciplinary programs."},
        {"name": "BITS Pilani", "exam": "BITSAT", "highlights": "Flexible curriculum, strong alumni network, and excellent global exposure."}
    ],
    "Medical & Healthcare": [
        {"name": "AIIMS New Delhi", "exam": "NEET-UG", "highlights": "Best-in-class MBBS program with extensive clinical exposure."},
        {"name": "CMC Vellore", "exam": "NEET-UG", "highlights": "Strong community medicine focus and affordable medical education."},
        {"name": "KMC Manipal", "exam": "NEET-UG", "highlights": "Modern infrastructure, research opportunities, and global recognition."}
    ],
    "Business & Management": [
        {"name": "IIM Ahmedabad (PGP)", "exam": "CAT", "highlights": "Premier management institute with stellar placements."},
        {"name": "IIM Bangalore (PGP)", "exam": "CAT", "highlights": "Leadership-focused curriculum and strong industry links."},
        {"name": "NMIMS Mumbai (BBA/MBA)", "exam": "NPAT / NMAT", "highlights": "Urban campus with great corporate exposure and entrepreneurship cell."}
    ],
    "Law & Legal Services": [
        {"name": "NLSIU Bengaluru", "exam": "CLAT", "highlights": "India's top law school with excellent moot court culture."},
        {"name": "NALSAR Hyderabad", "exam": "CLAT", "highlights": "Strong corporate law placements and international exchange."},
        {"name": "NLU Delhi", "exam": "AILET", "highlights": "Focus on policy, litigation, and research-driven curriculum."}
    ],
    "Design & Creative Arts": [
        {"name": "NID Ahmedabad", "exam": "NID DAT", "highlights": "Flagship design school known for product & industrial design."},
        {"name": "IIT Bombay (IDC)", "exam": "CEED / UCEED", "highlights": "Blend of engineering and design with innovative labs."},
        {"name": "NIFT Delhi", "exam": "NIFT Entrance", "highlights": "Top fashion & lifestyle design institute with strong industry ties."}
    ],
    "Data Science & Analytics": [
        {"name": "ISI Kolkata", "exam": "ISI Entrance", "highlights": "Premier statistics institute with rigorous analytics programs."},
        {"name": "IISc Bengaluru", "exam": "JEE / KVPY / GATE", "highlights": "Advanced research in AI/ML and interdisciplinary collaborations."},
        {"name": "IIT Hyderabad", "exam": "JEE Advanced", "highlights": "Dedicated AI programs and partnerships with tech giants."}
    ],
    "General": [
        {"name": "Delhi University (Top Colleges)", "exam": "CUET", "highlights": "Wide range of UG programs with vibrant campus culture."},
        {"name": "Christ University", "exam": "Institution Entrance", "highlights": "Strong holistic development and diverse course options."},
        {"name": "Symbiosis International University", "exam": "SET / SNAP", "highlights": "Modern campus, global curriculum, and active clubs."}
    ]
}

# Extend field and stream mappings with new clusters
FIELD_TO_STREAM.setdefault("Environmental Science", "Science")
FIELD_TO_STREAM.setdefault("Innovation & Future Tech", "Science")
for field_name, careers in FIELD_CAREER_CLUSTERS.items():
    FIELD_TO_STREAM.setdefault(field_name, FIELD_TO_STREAM.get(field_name, "Science" if field_name in SCIENCE_PATH_LOOKUP else None))
    for career in careers:
        FIELD_ROLE_MAP.setdefault(career, set()).add(field_name)
        parent_stream = FIELD_TO_STREAM.get(field_name)
        if parent_stream:
            STREAM_ROLE_MAP.setdefault(career, set()).add(parent_stream)


# ---------------- RESUME BUILDER CLASS ----------------
class ResumeBuilder:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the resume"""
        # Title Style
        self.styles.add(ParagraphStyle(
            name='ResumeTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Section Header Style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=20,
            leftIndent=0
        ))
        
        # Normal Text Style
        self.styles.add(ParagraphStyle(
            name='ResumeText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=6
        ))
        
        # Skill Style
        self.styles.add(ParagraphStyle(
            name='SkillText',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=3
        ))

    def create_resume(self, user_data, career_data, filename):
        """Create a professional resume PDF"""
        doc = SimpleDocTemplate(filename, pagesize=A4, 
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.5*inch, rightMargin=0.5*inch)
        
        story = []
        
        # Header Section
        story.extend(self.create_header_section(user_data))
        
        # Career Objective
        story.extend(self.create_objective_section(user_data, career_data))
        
        # Education
        story.extend(self.create_education_section(user_data, career_data))
        
        # Skills
        story.extend(self.create_skills_section(user_data, career_data))
        
        # Projects/Experience
        story.extend(self.create_experience_section(user_data, career_data))
        
        # Achievements
        story.extend(self.create_achievements_section(user_data))
        
        # Build PDF
        doc.build(story)
        
        return filename

    def create_header_section(self, user_data):
        """Create resume header section"""
        elements = []
        
        # Name
        name = user_data.get('name', 'Your Name')
        title = Paragraph(f"<b>{name}</b>", self.styles['ResumeTitle'])
        elements.append(title)
        
        # Contact Information
        contact_info = [
            user_data.get('email', 'your.email@example.com'),
            user_data.get('phone', '+91 XXXXXXXXXX'),
            user_data.get('location', 'Your City, State'),
            user_data.get('linkedin', 'linkedin.com/in/yourprofile')
        ]
        
        contact_text = " | ".join(filter(None, contact_info))
        contact_para = Paragraph(contact_text, self.styles['ResumeText'])
        elements.append(contact_para)
        elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def create_objective_section(self, user_data, career_data):
        """Create career objective section"""
        elements = []
        
        objective_text = self.generate_career_objective(user_data, career_data)
        section_header = Paragraph("<b>CAREER OBJECTIVE</b>", self.styles['SectionHeader'])
        objective_para = Paragraph(objective_text, self.styles['ResumeText'])
        
        elements.append(section_header)
        elements.append(objective_para)
        elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def create_education_section(self, user_data, career_data):
        """Create education section"""
        elements = []
        
        section_header = Paragraph("<b>EDUCATION</b>", self.styles['SectionHeader'])
        elements.append(section_header)
        
        # Current stream/education
        stream = user_data.get('stream', '')
        field = user_data.get('field', '')
        role = user_data.get('role', '')
        
        education_data = [
            ["2020-2024", f"Bachelor's in {field if field else 'Relevant Field'}", "University Name", "CGPA: 8.5/10"],
            ["2018-2020", f"12th Grade - {stream} Stream", "School Name", "Percentage: 85%"],
            ["2018", "10th Grade", "School Name", "Percentage: 90%"]
        ]
        
        # Adjust based on career recommendations
        recommended_career = career_data[0][0] if career_data else "Professional"
        if "Engineer" in recommended_career or "Data" in recommended_career:
            education_data[0][1] = "Bachelor's in Computer Science/Engineering"
        elif "Doctor" in recommended_career or "Medical" in recommended_career:
            education_data[0][1] = "MBBS/Bachelor's in Medical Sciences"
        elif "Business" in recommended_career or "Manager" in recommended_career:
            education_data[0][1] = "Bachelor's in Business Administration"
        
        table = Table(education_data, colWidths=[1.2*inch, 2.5*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def create_skills_section(self, user_data, career_data):
        """Create skills section"""
        elements = []
        
        section_header = Paragraph("<b>SKILLS & COMPETENCIES</b>", self.styles['SectionHeader'])
        elements.append(section_header)
        
        # Get skills from user interests and career data
        skills = self.generate_skills(user_data, career_data)
        
        # Create two-column layout for skills
        skill_table_data = []
        mid_point = len(skills) // 2 + len(skills) % 2
        
        for i in range(mid_point):
            row = []
            if i < len(skills):
                row.append(f"• {skills[i]}")
            if i + mid_point < len(skills):
                row.append(f"• {skills[i + mid_point]}")
            else:
                row.append("")
            skill_table_data.append(row)
        
        if skill_table_data:
            table = Table(skill_table_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ]))
            elements.append(table)
        
        elements.append(Spacer(1, 0.1*inch))
        return elements

    def create_experience_section(self, user_data, career_data):
        """Create projects/experience section"""
        elements = []
        
        section_header = Paragraph("<b>PROJECTS & EXPERIENCE</b>", self.styles['SectionHeader'])
        elements.append(section_header)
        
        projects = self.generate_projects(user_data, career_data)
        
        for project in projects:
            project_text = f"<b>{project['title']}</b> - {project['duration']}<br/>{project['description']}"
            project_para = Paragraph(project_text, self.styles['ResumeText'])
            elements.append(project_para)
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.1*inch))
        return elements

    def create_achievements_section(self, user_data):
        """Create achievements section"""
        elements = []
        
        section_header = Paragraph("<b>ACHIEVEMENTS & CERTIFICATIONS</b>", self.styles['SectionHeader'])
        elements.append(section_header)
        
        achievements = [
            "Academic Excellence Scholarship 2022",
            "1st Prize in Inter-College Technical Fest",
            "Certified in Python Programming",
            "Volunteer of the Year - Social Service Club"
        ]
        
        for achievement in achievements:
            achievement_para = Paragraph(f"• {achievement}", self.styles['ResumeText'])
            elements.append(achievement_para)
        
        return elements

    def generate_career_objective(self, user_data, career_data):
        """Generate personalized career objective"""
        stream = user_data.get('stream', '')
        interests = user_data.get('interests', '')
        role = user_data.get('role', '')
        career = career_data[0][0] if career_data else "professional"
        
        objectives = {
            "Software Engineer": f"A motivated {stream} student with strong interest in {interests}. Seeking a Software Engineer position to apply programming skills and contribute to innovative software solutions.",
            "Data Scientist": f"Analytical-minded {stream} graduate passionate about {interests}. Looking for a Data Scientist role to leverage statistical analysis and machine learning for data-driven insights.",
            "Doctor": f"Dedicated {stream} student with deep interest in healthcare. Aspiring to become a {role} to provide quality medical care and contribute to patient well-being.",
            "Business Manager": f"Dynamic {stream} graduate with leadership qualities and interest in {interests}. Seeking Business Manager position to drive organizational growth and operational excellence."
        }
        
        return objectives.get(career, f"Enthusiastic {stream} student seeking a {career} position to apply academic knowledge and grow professionally.")

    def generate_skills(self, user_data, career_data):
        """Generate relevant skills based on user profile"""
        base_skills = ["Communication", "Problem Solving", "Teamwork", "Time Management"]
        
        # Add skills based on interests
        hobby = user_data.get('hobby', '')
        free_time = user_data.get('free_time', '')
        subject = user_data.get('interested_subject', '')
        
        interest_skills = []
        if "Coding" in hobby or "Programming" in hobby:
            interest_skills.extend(["Python", "Java", "Algorithms", "Debugging"])
        if "Design" in hobby:
            interest_skills.extend(["UI/UX Design", "Creative Thinking", "Adobe Suite"])
        if "Finance" in hobby:
            interest_skills.extend(["Financial Analysis", "Excel", "Market Research"])
        if "Research" in hobby:
            interest_skills.extend(["Data Analysis", "Research Methodology", "Report Writing"])
        
        # Add career-specific skills
        career = career_data[0][0] if career_data else ""
        career_skills = {
            "Software Engineer": ["Python/Java/C++", "Data Structures", "OOP", "Git", "SQL", "Agile Methodology"],
            "Data Scientist": ["Machine Learning", "Statistics", "Data Visualization", "SQL", "Python/R", "Pandas"],
            "Doctor": ["Patient Care", "Medical Knowledge", "Diagnosis", "Emergency Handling", "Communication"],
            "Business Manager": ["Leadership", "Strategic Planning", "Project Management", "Budgeting", "Team Management"]
        }
        
        return base_skills + interest_skills + career_skills.get(career, [])

    def generate_projects(self, user_data, career_data):
        """Generate relevant projects based on interests and career"""
        career = career_data[0][0] if career_data else ""
        hobby = user_data.get('hobby', '')
        
        projects = {
            "Software Engineer": [
                {
                    "title": "E-commerce Website Development",
                    "duration": "Jan 2023 - Mar 2023",
                    "description": "Developed a full-stack e-commerce platform using React and Node.js with user authentication and payment integration."
                },
                {
                    "title": "Mobile App for Task Management",
                    "duration": "Sep 2022 - Dec 2022", 
                    "description": "Created a cross-platform mobile application using Flutter for personal task management with cloud synchronization."
                }
            ],
            "Data Scientist": [
                {
                    "title": "Customer Segmentation Analysis",
                    "duration": "Feb 2023 - Apr 2023",
                    "description": "Implemented K-means clustering for customer segmentation using Python and scikit-learn, improving marketing strategy."
                },
                {
                    "title": "Sales Prediction Model",
                    "duration": "Oct 2022 - Jan 2023",
                    "description": "Built a machine learning model to predict sales using historical data, achieving 85% accuracy."
                }
            ],
            "Doctor": [
                {
                    "title": "Medical Internship",
                    "duration": "Jun 2023 - Aug 2023", 
                    "description": "Completed 200+ hours of clinical observation, assisted in patient care and medical procedures."
                },
                {
                    "title": "Health Awareness Campaign",
                    "duration": "Mar 2023 - May 2023",
                    "description": "Organized community health awareness program reaching 500+ people on preventive healthcare."
                }
            ],
            "Business Manager": [
                {
                    "title": "Business Plan Development",
                    "duration": "Jan 2023 - Mar 2023",
                    "description": "Created comprehensive business plan for startup including market analysis, financial projections, and operational strategy."
                },
                {
                    "title": "Team Leadership Project", 
                    "duration": "Sep 2022 - Dec 2022",
                    "description": "Led a team of 5 members in organizing college fest, managing budget of ₹2 lakhs and coordinating 20+ events."
                }
            ]
        }
        
        return projects.get(career, [
            {
                "title": "Academic Project",
                "duration": "2022-2023", 
                "description": "Completed comprehensive project demonstrating skills and knowledge in chosen field of study."
            }
        ])


# ---------------- PERSONAL INFO DIALOG ----------------
class PersonalInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Personal Information for Resume")
        self.setModal(True)
        self.setFixedSize(500, 400)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3c3c3c, stop:1 #202020);
                border-radius: 15px;
            }
            QLineEdit, QComboBox {
                background: rgba(20, 20, 20, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 10px;
                color: #f5f5f5;
                font-size: 14px;
            }
            QLabel {
                color: #f5f5f5;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📝 Personal Information for Resume")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5f5f5; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Form layout
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        # Personal information fields
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your full name")
        
        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("your.email@example.com")
        
        self.phone_edit = QLineEdit()
        self.phone_edit.setPlaceholderText("+91 XXXXXXXXXX")
        
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("City, State")
        
        self.linkedin_edit = QLineEdit()
        self.linkedin_edit.setPlaceholderText("linkedin.com/in/yourprofile")
        
        # Add fields to form
        form_layout.addRow("Full Name:", self.name_edit)
        form_layout.addRow("Email:", self.email_edit)
        form_layout.addRow("Phone:", self.phone_edit)
        form_layout.addRow("Location:", self.location_edit)
        form_layout.addRow("LinkedIn:", self.linkedin_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        generate_btn = QPushButton("Generate Resume PDF")
        generate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #34d399);
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #10b981);
            }
        """)
        generate_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(generate_btn)
        
        layout.addLayout(button_layout)
        
    def get_personal_info(self):
        """Get the entered personal information"""
        return {
            'name': self.name_edit.text().strip(),
            'email': self.email_edit.text().strip(),
            'phone': self.phone_edit.text().strip(),
            'location': self.location_edit.text().strip(),
            'linkedin': self.linkedin_edit.text().strip()
        }


# ---------------- RESPONSIVE MAIN APP ----------------
class CareerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Career Guidance System - Modern Glass UI")
        self.resize(1200, 800)
        self.setStyleSheet(APP_STYLE)
        self.setMinimumSize(1000, 700)
        self._active_animations = []

        # Initialize data
        self.initialize_data()
        
        # Initialize resume builder
        self.resume_builder = ResumeBuilder()

        # Create main layout
        container = QWidget()
        container.setObjectName("rootContainer")
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(0)

        # Create stacked widget for different views
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.stack)

        # Build different pages
        self.build_input_page()
        self.build_results_page()
        
        # Show input page first
        self.stack.setCurrentWidget(self.input_page)
        
        # Store current recommendations
        self.current_recommendations = []

    def initialize_data(self):
        """Initialize career data and models"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, "model")

        # Load ML model
        self.model = None
        try:
            if os.path.exists(os.path.join(model_dir, "career_model.pkl")):
                self.model = joblib.load(os.path.join(model_dir, "career_model.pkl"))
                print("ML model loaded successfully")
        except Exception as e:
            print(f"Model loading warning: {e}")

        # Load career details - merge JSON file with ENHANCED_CAREER_DETAILS
        # JSON should only fill gaps, ENHANCED_CAREER_DETAILS takes priority
        json_details = safe_load_json(os.path.join(model_dir, "career_details.json"), {})
        self.career_details = {}
        # Start with ENHANCED_CAREER_DETAILS
        for career, details in ENHANCED_CAREER_DETAILS.items():
            self.career_details[career] = details.copy()
        # Merge JSON details, but preserve roadmap, pros, cons from ENHANCED if they exist
        for career, json_data in json_details.items():
            if career in self.career_details:
                # Merge, but keep roadmap, pros, cons from ENHANCED if present
                merged = self.career_details[career].copy()
                for key, value in json_data.items():
                    if key not in ["roadmap", "pros", "cons", "sub_specialty_steps"] or key not in merged:
                        merged[key] = value
                self.career_details[career] = merged
            else:
                self.career_details[career] = json_data
        
        # Ensure all careers from CAREER_MAPPINGS have at least basic profiles
        all_careers = set()
        for field_roles in CAREER_MAPPINGS["roles"].values():
            all_careers.update(field_roles)
        
        # Add default profiles for missing careers
        for career in all_careers:
            if career not in self.career_details:
                # Get field info for context
                fields = FIELD_ROLE_MAP.get(career, set())
                field_name = list(fields)[0] if fields else "General"
                stream = FIELD_TO_STREAM.get(field_name, "General")
                
                self.career_details[career] = {
                    "description": f"{career} professional working in {field_name} sector.",
                    "education": ["Relevant Bachelor's Degree", "Industry Certifications"],
                    "skills": ["Communication", "Problem-solving", "Industry-specific skills"],
                    "salary": "₹5-20 LPA (Varies by experience and location)",
                    "market": "Growing demand in relevant sectors.",
                    "pros": ["Career growth opportunities", "Diverse work environment", "Industry-specific benefits"],
                    "cons": ["Competitive field", "Need continuous learning", "Industry-specific challenges"],
                    "roadmap": [
                        f"Complete 12th in relevant stream",
                        f"Pursue relevant bachelor's degree in {field_name}",
                        "Gain industry experience through internships",
                        "Obtain relevant certifications",
                        "Start entry-level position",
                        "Progress to senior roles with experience"
                    ]
                }
            else:
                # Ensure existing careers have roadmap if missing
                if "roadmap" not in self.career_details[career] and "sub_specialty_steps" not in self.career_details[career]:
                    self.career_details[career]["roadmap"] = [
                        f"Complete 12th in relevant stream",
                        f"Pursue relevant education for {career}",
                        "Gain practical experience",
                        "Obtain necessary certifications/licenses",
                        "Start career in entry-level position",
                        "Progress with experience and skills"
                    ]

        # Load embedding model
        try:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load embedding model:\n{e}")
            sys.exit(1)

        # Precompute embeddings for all careers
        self.embeddings = {}
        # First, add all careers from career_details
        for job, details in self.career_details.items():
            text = details.get("description", job)
            self.embeddings[job] = self.embed_model.encode(text, convert_to_tensor=True)
        
        # Add embeddings for all careers in CAREER_MAPPINGS that might not be in career_details
        all_careers = set()
        for field_roles in CAREER_MAPPINGS["roles"].values():
            all_careers.update(field_roles)
        
        for career in all_careers:
            if career not in self.embeddings:
                # Create a basic description if not in career_details
                if career in self.career_details:
                    text = self.career_details[career].get("description", career)
                else:
                    fields = FIELD_ROLE_MAP.get(career, set())
                    field_name = list(fields)[0] if fields else "General"
                    text = f"{career} professional working in {field_name} sector."
                self.embeddings[career] = self.embed_model.encode(text, convert_to_tensor=True)

    def build_input_page(self):
        """Build the responsive input page"""
        self.input_page = QWidget()
        self.input_page.setObjectName("contentPage")
        layout = QVBoxLayout(self.input_page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header with responsive sizing
        header_frame = QFrame()
        header_frame.setObjectName("glassFrame")
        header_frame.setMaximumHeight(150)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel("🎯 AI Career Guidance System")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        subtitle = QLabel("Discover Your Perfect Career Path with AI")
        subtitle.setStyleSheet("font-size: clamp(14px, 2vw, 16px); color: #94a3b8; text-align: center; padding: 5px;")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_frame)
        self.animate_widget_entry(header_frame, delay=120, distance=35)
        self.animate_widget_entry(header_frame, delay=200, distance=40)

        # Main input container
        input_container = QFrame()
        input_container.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_container)
        input_layout.setSpacing(20)
        input_layout.setContentsMargins(25, 25, 25, 25)

        # Use grid layout for better responsiveness
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(15)
        grid_layout.setHorizontalSpacing(15)

        self.inputs = {}
        self.stream_cb = None
        self.science_track_cb = None
        self.science_track_label = None
        self.field_cb = None
        self.role_cb = None
        
        # Input fields configuration
        input_config = [
            (0, "📘 Educational Stream", "stream", CAREER_MAPPINGS["streams"]),
            (1, "🧬 Science Focus (Medical / Non-Medical)", "science_focus", []),
            (2, "🎯 Career Field", "field", []),
            (3, "🧠 Preferred Role", "role", []),
            (4, "🎮 Primary Interest", "hobby", HOBBY_OPTIONS),
            (5, "⏳ Free Time Activity", "free_time", FREE_TIME_OPTIONS),
            (6, "📚 Favorite Subject", "interested_subject", SUBJECT_OPTIONS),
        ]

        for row, label_text, key, options in input_config:
            # Label
            label = QLabel(label_text)
            label.setObjectName("sectionTitle")
            label.setWordWrap(True)
            grid_layout.addWidget(label, row, 0)

            # ComboBox
            combo = QComboBox()
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.setMinimumHeight(35)

            if key == "stream":
                self.stream_cb = combo
                combo.addItems(options)
                combo.currentTextChanged.connect(self.update_fields)
            elif key == "science_focus":
                self.science_track_label = label
                self.science_track_cb = combo
                combo.addItem("Select focus", None)
                combo.addItem("Medical (Biology)", "Medical")
                combo.addItem("Non-Medical (Maths)", "Non-Medical")
                combo.currentTextChanged.connect(self.update_fields)
                label.setVisible(False)
                combo.setVisible(False)
            elif key == "field":
                self.field_cb = combo
                combo.currentTextChanged.connect(self.update_roles)
            elif key == "role":
                self.role_cb = combo

            if key not in {"science_focus"}:
                self.inputs[key] = combo

            if key in {"field", "role", "science_focus", "stream"}:
                # handled separately or populated dynamically
                pass
            else:
                combo.addItems(options)

            grid_layout.addWidget(combo, row, 1)

        input_layout.addLayout(grid_layout)
        self.update_fields()

        # Description input
        desc_label = QLabel("💭 Tell us about your aspirations:")
        desc_label.setObjectName("sectionTitle")
        input_layout.addWidget(desc_label)
        
        self.free_text = QTextEdit()
        self.free_text.setPlaceholderText("I enjoy solving complex problems, working with technology, and have strong analytical skills. I'm passionate about innovation and want to make a positive impact...")
        self.free_text.setMaximumHeight(120)
        self.free_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.free_text)

        # Submit button container for centering
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        submit_btn = QPushButton("🚀 Get Career Recommendations")
        submit_btn.setMinimumHeight(45)
        submit_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        submit_btn.clicked.connect(self.run_prediction)
        self.apply_button_glow(submit_btn, color="#60a5fa")
        
        button_layout.addStretch()
        button_layout.addWidget(submit_btn)
        button_layout.addStretch()
        
        input_layout.addWidget(button_container)
        layout.addWidget(input_container)
        self.animate_widget_entry(input_container, delay=350, distance=40)
        layout.addStretch()

        self.stack.addWidget(self.input_page)

    def build_results_page(self):
        """Build the responsive results page"""
        self.results_page = QWidget()
        self.results_page.setObjectName("contentPage")
        layout = QVBoxLayout(self.results_page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Header with back button
        header_widget = QWidget()
        header_widget.setMaximumHeight(60)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        back_btn = QPushButton("← Back")
        back_btn.setObjectName("secondary")
        back_btn.setFixedSize(120, 40)
        back_btn.clicked.connect(lambda: self.transition_to_page(self.input_page))
        
        title = QLabel("Your Career Recommendations")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(QWidget())  # Spacer for balance
        
        layout.addWidget(header_widget)
        self.animate_widget_entry(header_widget, delay=180, distance=35)

        # Results tabs
        self.results_tabs = QTabWidget()
        self.results_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Summary Tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_scroll = QScrollArea()
        self.summary_scroll.setWidgetResizable(True)
        self.summary_content = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_content)
        self.summary_scroll.setWidget(self.summary_content)
        summary_layout.addWidget(self.summary_scroll)
        self.results_tabs.addTab(self.summary_tab, "📊 Overview")

        # Details Tab
        self.details_tab = QWidget()
        details_layout = QVBoxLayout(self.details_tab)
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_content = QWidget()
        self.details_layout = QVBoxLayout(self.details_content)
        self.details_scroll.setWidget(self.details_content)
        details_layout.addWidget(self.details_scroll)
        self.results_tabs.addTab(self.details_tab, "📋 Career Details")

        # Analytics Tab
        self.analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(self.analytics_tab)
        self.analytics_scroll = QScrollArea()
        self.analytics_scroll.setWidgetResizable(True)
        self.analytics_content = QWidget()
        self.analytics_layout = QVBoxLayout(self.analytics_content)
        self.analytics_scroll.setWidget(self.analytics_content)
        analytics_layout.addWidget(self.analytics_scroll)
        self.results_tabs.addTab(self.analytics_tab, "📈 Analytics")
        
        # Resume Builder Tab
        self.resume_tab = QWidget()
        resume_layout = QVBoxLayout(self.resume_tab)
        self.resume_scroll = QScrollArea()
        self.resume_scroll.setWidgetResizable(True)
        self.resume_content = QWidget()
        self.resume_layout = QVBoxLayout(self.resume_content)
        self.resume_scroll.setWidget(self.resume_content)
        resume_layout.addWidget(self.resume_scroll)
        self.results_tabs.addTab(self.resume_tab, "📄 Build Resume")

        layout.addWidget(self.results_tabs)
        self.stack.addWidget(self.results_page)

    def update_fields(self):
        """Update field options based on stream selection and science focus"""
        if not self.stream_cb:
            return

        stream = self.stream_cb.currentText()
        is_science = stream == "Science"

        if hasattr(self, "science_track_label"):
            self.science_track_label.setVisible(is_science)
        if hasattr(self, "science_track_cb"):
            if is_science:
                self.science_track_cb.setVisible(True)
            else:
                self.science_track_cb.setVisible(False)
                if self.science_track_cb.currentIndex() != 0:
                    self.science_track_cb.setCurrentIndex(0)

        if not self.field_cb:
            return

        self.field_cb.blockSignals(True)
        self.field_cb.clear()

        if is_science:
            focus = self.get_science_focus()
            allowed_fields = get_science_path_labels_for_focus(focus)
            if not allowed_fields:
                allowed_fields = ["Select science focus above"]
        else:
            allowed_fields = CAREER_MAPPINGS["fields"].get(stream, [])

        if allowed_fields:
            self.field_cb.addItems(allowed_fields)
        else:
            self.field_cb.addItems(["Select a stream first"])

        self.field_cb.blockSignals(False)
        self.update_roles()

    def update_roles(self):
        """Update role options based on field selection"""
        if not self.field_cb or not self.role_cb:
            return

        field = self.field_cb.currentText()
        self.role_cb.blockSignals(True)
        self.role_cb.clear()
        if field in CAREER_MAPPINGS["roles"]:
            self.role_cb.addItems(CAREER_MAPPINGS["roles"][field])
        self.role_cb.blockSignals(False)

    def get_science_focus(self):
        """Return the selected science focus (Medical/Non-Medical) if applicable."""
        if not hasattr(self, "science_track_cb"):
            return None
        if self.stream_cb.currentText() != "Science":
            return None
        return self.science_track_cb.currentData()

    def get_science_focus_label(self):
        focus = self.get_science_focus()
        if focus and hasattr(self, "science_track_cb"):
            return self.science_track_cb.currentText()
        return ""

    def run_prediction(self):
        """Run career prediction and show results"""
        try:
            # Clear previous results
            self.clear_layout(self.summary_layout)
            self.clear_layout(self.details_layout)
            self.clear_layout(self.analytics_layout)
            self.clear_layout(self.resume_layout)

            # Get user inputs
            profile_parts = [
                f"Stream: {self.stream_cb.currentText()}",
                f"Field: {self.field_cb.currentText()}",
                f"Role: {self.role_cb.currentText()}",
                f"Hobby: {self.inputs['hobby'].currentText()}",
                f"Free time: {self.inputs['free_time'].currentText()}",
                f"Interest: {self.inputs['interested_subject'].currentText()}",
                self.free_text.toPlainText().strip()
            ]

            science_focus_label = self.get_science_focus_label()
            if science_focus_label:
                profile_parts.append(f"Science focus: {science_focus_label}")

            user_profile_text = " ".join(filter(None, profile_parts))

            # Get recommendations
            self.current_recommendations = self.get_career_recommendations(user_profile_text)

            # Display results
            self.display_summary(self.current_recommendations)
            self.display_details(self.current_recommendations)
            self.display_analytics(self.current_recommendations)
            self.display_resume_builder()

            # Switch to results page
            self.transition_to_page(self.results_page)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{str(e)}")

    def get_career_recommendations(self, user_text):
        """Get career recommendations based on user input with proper filtering"""
        # Get semantic similarity scores
        if user_text.strip():
            user_emb = self.embed_model.encode(user_text, convert_to_tensor=True)
            scores = {}
            
            for job, emb in self.embeddings.items():
                try:
                    scores[job] = float(util.cos_sim(user_emb, emb))
                except:
                    scores[job] = 0.0
        else:
            scores = {job: 0.0 for job in self.embeddings.keys()}

        # Pre-filter by stream before ranking to ensure relevance
        selected_stream = self.stream_cb.currentText() if self.stream_cb else None
        focus = self.get_science_focus()
        
        # Filter scores by stream and science focus first
        filtered_scores = {}
        for job, score in scores.items():
            if selected_stream and selected_stream != "Other":
                if not self.is_career_valid_for_stream(job, selected_stream):
                    continue
            if focus and not self.is_career_valid_for_science_focus(job, focus):
                continue
            filtered_scores[job] = score

        # If no filtered results, use fallbacks
        if not filtered_scores or all(score == 0 for score in filtered_scores.values()):
            final = self.get_field_fallbacks()
        else:
            # Get top 6-8 recommendations for better variety
            final = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:8]

        # Expand related careers and prioritize
        final = self.expand_related_careers(final)
        final = self.prioritize_recommendations_by_field(final)
        
        # Final stream filter (redundant but ensures correctness)
        final = self.filter_recommendations_by_stream(final)
        
        # Return top 4-6 most relevant
        return final[:6] if len(final) > 6 else final

    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def prioritize_recommendations_by_field(self, recommendations):
        """Boost and reorder careers that align with the selected field."""
        preferred_fields = self.get_preferred_fields()
        science_focus = self.get_science_focus()
        if not preferred_fields and not science_focus:
            return recommendations

        interest_fields = self.get_interest_fields()

        prioritized = []
        for career, score in recommendations:
            bonus = 0.0
            career_fields = FIELD_ROLE_MAP.get(career, set())
            if preferred_fields and career_fields & preferred_fields:
                bonus = 0.4  # strong preference for matching field
            elif self.stream_cb.currentText() in STREAM_ROLE_MAP.get(career, set()):
                bonus = 0.1  # mild boost for matching stream
            if science_focus and self.is_career_valid_for_science_focus(career, science_focus):
                bonus = max(bonus, 0.25)
            if interest_fields and career_fields & interest_fields:
                bonus = max(bonus, 0.3)

            prioritized.append((career, min(score + bonus, 1.0)))

        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized

    def get_related_career_targets(self):
        targets = set()

        selected_field = self.field_cb.currentText() if self.field_cb else ""
        targets.update(self.get_cluster_roles(selected_field))

        for field in self.get_subject_fields():
            targets.update(self.get_cluster_roles(field))

        for field in self.get_interest_fields():
            targets.update(self.get_cluster_roles(field))

        if self.stream_cb.currentText() == "Science":
            focus = self.get_science_focus()
            for label in get_science_path_labels_for_focus(focus):
                targets.update(self.get_cluster_roles(label))

        return targets

    def expand_related_careers(self, recommendations):
        related = self.get_related_career_targets()
        if not related:
            return recommendations

        rec_dict = {career: score for career, score in recommendations}
        max_score = max(rec_dict.values(), default=0.7)
        base_score = max_score * 0.85 if max_score > 0 else 0.6

        for career in related:
            if career in rec_dict:
                rec_dict[career] = min(rec_dict[career] + 0.05, 1.0)
            else:
                rec_dict[career] = base_score

        expanded = sorted(rec_dict.items(), key=lambda x: x[1], reverse=True)
        return expanded[: max(6, len(recommendations))]

    def filter_recommendations_by_stream(self, recommendations):
        """Keep careers that belong to the user's selected stream."""
        selected_stream = self.stream_cb.currentText()
        if not selected_stream or selected_stream == "Other":
            return recommendations

        focus = self.get_science_focus()
        filtered = []
        for career, score in recommendations:
            if not self.is_career_valid_for_stream(career, selected_stream):
                continue
            if focus and not self.is_career_valid_for_science_focus(career, focus):
                continue
            filtered.append((career, score))

        if filtered:
            return filtered

        return self.get_stream_fallbacks(selected_stream)

    def is_career_valid_for_stream(self, career, stream):
        """Determine whether a career is mapped to the given stream."""
        allowed_streams = STREAM_ROLE_MAP.get(career)
        if not allowed_streams:
            return True
        return stream in allowed_streams

    def is_career_valid_for_science_focus(self, career, focus=None):
        """Ensure science recommendations align with the Medical/Non-Medical choice."""
        focus = focus or self.get_science_focus()
        if not focus:
            return True
        allowed_fields = get_science_field_tags_for_focus(focus)
        if not allowed_fields:
            return True
        return bool(FIELD_ROLE_MAP.get(career, set()) & allowed_fields)

    def get_stream_fallbacks(self, stream):
        """Return fallback roles that align with the current stream."""
        roles = []
        focus = self.get_science_focus()
        if stream == "Science":
            field_list = get_science_path_labels_for_focus(focus)
            if not field_list:
                field_list = CAREER_MAPPINGS["fields"]["Science"]
        else:
            field_list = CAREER_MAPPINGS["fields"].get(stream, [])

        for field in field_list:
            cluster_roles = FIELD_CAREER_CLUSTERS.get(field)
            if cluster_roles:
                for role in cluster_roles:
                    if role not in roles:
                        roles.append(role)
                continue
            for role in CAREER_MAPPINGS["roles"].get(field, []):
                if role not in roles:
                    roles.append(role)

        if not roles:
            roles = ["Software Engineer", "Data Scientist", "Doctor", "Business Manager"]

        return [(role, 0.6) for role in roles[:4]]

    def get_field_fallbacks(self):
        """Fallback careers derived from the currently selected field."""
        selected_field = self.field_cb.currentText()
        cluster_roles = self.get_cluster_roles(selected_field)
        if cluster_roles:
            return [(role, 0.65) for role in cluster_roles[:4]]

        if self.stream_cb.currentText() == "Science":
            focus = self.get_science_focus()
            roles = []
            for label in get_science_path_labels_for_focus(focus):
                roles.extend(self.get_cluster_roles(label))
            if roles:
                return [(role, 0.65) for role in roles[:4]]

        subject_fields = self.get_subject_fields()
        if subject_fields:
            roles = []
            for field in subject_fields:
                roles.extend(self.get_cluster_roles(field))
            if roles:
                return [(role, 0.65) for role in roles[:4]]

        interest_fields = self.get_interest_fields()
        if interest_fields:
            roles = []
            for field in interest_fields:
                roles.extend(self.get_cluster_roles(field))
            if roles:
                return [(role, 0.65) for role in roles[:4]]

        # default mix if no field selected
        return [("Software Engineer", 0.8), ("Data Scientist", 0.7),
                ("Doctor", 0.6), ("Business Manager", 0.5)]

    def get_subject_fields(self):
        subject_combo = self.inputs.get("interested_subject")
        if not subject_combo:
            return set()
        return set(SUBJECT_FIELD_MAP.get(subject_combo.currentText(), []))

    def get_preferred_fields(self):
        fields = set()
        if self.field_cb:
            field_value = self.field_cb.currentText()
            if field_value and field_value not in {"Select a stream first"}:
                fields.add(field_value)
        subject_fields = self.get_subject_fields()
        fields.update(subject_fields)
        interest_fields = self.get_interest_fields()
        fields.update(interest_fields)
        fields.update(self.get_science_focus_field_tags())
        return fields

    def get_interest_fields(self):
        hobby_combo = self.inputs.get("hobby")
        if not hobby_combo:
            return set()
        return set(INTEREST_FIELD_MAP.get(hobby_combo.currentText(), []))

    def get_science_focus_field_tags(self):
        focus = self.get_science_focus()
        if not focus:
            return set()
        return get_science_field_tags_for_focus(focus)

    def get_cluster_roles(self, field_name):
        if not field_name or field_name in {"Select science focus above", "Select a stream first"}:
            return []
        if field_name in FIELD_CAREER_CLUSTERS:
            return FIELD_CAREER_CLUSTERS[field_name]
        return CAREER_MAPPINGS["roles"].get(field_name, [])

    def get_college_info(self, career):
        """Return curated college info for the given career."""
        fields = FIELD_ROLE_MAP.get(career, set())
        for field in fields:
            if field in COLLEGE_INFO_BY_FIELD:
                return COLLEGE_INFO_BY_FIELD[field]
        return COLLEGE_INFO_BY_FIELD.get("General", [])

    def display_summary(self, recommendations):
        """Display career recommendations summary"""
        # Header
        header = QLabel("🎯 Your Top Career Matches")
        header.setStyleSheet("""
            font-size: clamp(20px, 3vw, 24px); 
            font-weight: bold; 
            color: #60a5fa; 
            padding: 15px; 
            text-align: center;
        """)
        self.summary_layout.addWidget(header)

        # Recommendation cards in a scrollable layout
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setSpacing(20)
        cards_layout.setContentsMargins(12, 12, 12, 12)

        for i, (job, score) in enumerate(recommendations, 1):
            card = self.create_career_card(job, score, i)
            cards_layout.addWidget(card)
            self.animate_widget_entry(card, delay=80 * i, distance=25)

        cards_layout.addStretch()
        
        # Add container to scroll area
        self.summary_layout.addWidget(cards_container)

    def create_career_card(self, job, score, rank):
        """Create a responsive career recommendation card"""
        card = QFrame()
        card.setObjectName("resultCard")
        card.setMinimumHeight(120)
        card.setMaximumHeight(180)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # Rank badge - fixed size
        rank_frame = QFrame()
        rank_frame.setFixedSize(60, 60)
        rank_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f59e0b, stop:1 #fbbf24);
                border-radius: 30px;
            }
        """)
        rank_layout = QVBoxLayout(rank_frame)
        rank_label = QLabel(f"#{rank}")
        rank_label.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: white; 
            text-align: center;
        """)
        rank_layout.addWidget(rank_label)
        
        score_label = QLabel(f"{score*100:.1f}%")
        score_label.setStyleSheet("""
            font-size: 11px; 
            color: white; 
            text-align: center;
        """)
        rank_layout.addWidget(score_label)

        layout.addWidget(rank_frame)

        # Career info - flexible
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(8)
        
        # Job title
        title = QLabel(job)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        info_layout.addWidget(title)
        
        # Description
        desc = QLabel(self.career_details.get(job, {}).get('description', 'Career description'))
        desc.setStyleSheet("font-size: 12px; color: #cbd5e1; line-height: 1.3;")
        desc.setWordWrap(True)
        info_layout.addWidget(desc)
        
        # Quick stats - horizontal layout
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setSpacing(8)
        
        salary = self.career_details.get(job, {}).get('salary', '₹5-20 LPA')
        education = self.career_details.get(job, {}).get('education', ['Various'])[0]
        
        stats_layout.addWidget(self.create_stat_badge("💰", salary))
        stats_layout.addWidget(self.create_stat_badge("🎓", education))
        stats_layout.addStretch()
        
        info_layout.addWidget(stats_widget)
        layout.addWidget(info_widget, 1)

        return card

    def create_stat_badge(self, icon, text):
        """Create a responsive stat badge"""
        badge = QLabel(f"{icon} {text}")
        badge.setStyleSheet("""
            background: rgba(59, 130, 246, 0.3);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 10px;
            color: #bfdbfe;
        """)
        badge.setWordWrap(True)
        return badge

    def display_details(self, recommendations):
        """Display detailed career information"""
        details_container = QWidget()
        details_main_layout = QVBoxLayout(details_container)
        details_main_layout.setSpacing(15)
        details_main_layout.setContentsMargins(10, 10, 10, 10)

        for idx, (job, score) in enumerate(recommendations, 1):
            # Create detailed career section
            section = QFrame()
            section.setObjectName("glassFrame")
            section_layout = QVBoxLayout(section)
            section_layout.setSpacing(12)
            section_layout.setContentsMargins(20, 15, 20, 15)

            # Header
            header = QLabel(f"🎯 {job} - Career Details")
            header.setStyleSheet("font-size: clamp(16px, 2.5vw, 18px); font-weight: bold; color: #fbbf24;")
            header.setWordWrap(True)
            section_layout.addWidget(header)

            details = self.career_details.get(job, {})
            
            button_row = QWidget()
            button_layout = QHBoxLayout(button_row)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.setSpacing(10)
            buttons_added = False

            if self.has_roadmap_content(details):
                roadmap_btn = QPushButton("🗺️ View Roadmap")
                roadmap_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3b82f6;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        padding: 12px 24px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #2563eb;
                    }
                    QPushButton:pressed {
                        background-color: #1d4ed8;
                    }
                """)
                roadmap_btn.clicked.connect(lambda checked, j=job, d=details: self.show_roadmap_dialog(j, d))
                button_layout.addWidget(roadmap_btn)
                buttons_added = True

                college_btn = QPushButton("🏫 Colleges Information")
                college_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2c2c2c;
                        color: #f5f5f5;
                        border: 1px solid rgba(255, 255, 255, 0.15);
                        border-radius: 8px;
                        padding: 12px 24px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3a3a3a;
                    }
                """)
                college_btn.clicked.connect(lambda checked, j=job: self.show_colleges_dialog(j))
                button_layout.addWidget(college_btn)
                buttons_added = True

            if buttons_added:
                button_layout.addStretch()
                section_layout.addWidget(button_row)
            
            # Use responsive grid for details
            details_grid = QWidget()
            grid_layout = QHBoxLayout(details_grid)
            grid_layout.setSpacing(15)
            
            # Left column
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setSpacing(10)
            
            info_sections = [
                ("📝 Description", details.get('description', 'No description available')),
                ("🎓 Education", ", ".join(details.get('education', ['Various']))),
                ("🛠️ Skills", ", ".join(details.get('skills', ['Various']))),
            ]
            
            for icon, content in info_sections:
                left_layout.addWidget(self.create_detail_item(icon, content))
            
            # Right column
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_layout.setSpacing(10)
            
            right_sections = [
                ("💰 Salary", details.get('salary', '₹5-20 LPA')),
                ("📈 Market", details.get('market', 'Growing demand')),
            ]
            
            for icon, content in right_sections:
                right_layout.addWidget(self.create_detail_item(icon, content))
            
            # Pros and Cons - stack vertically on small screens
            if 'pros' in details or 'cons' in details:
                pros_cons_widget = QWidget()
                pros_cons_layout = QVBoxLayout(pros_cons_widget)
                pros_cons_layout.setSpacing(10)
                
                pros = self.create_pros_cons_section("✅ Advantages", details.get('pros', []), "#10b981")
                cons = self.create_pros_cons_section("❌ Challenges", details.get('cons', []), "#ef4444")
                
                pros_cons_layout.addWidget(pros)
                pros_cons_layout.addWidget(cons)
                right_layout.addWidget(pros_cons_widget)
            
            grid_layout.addWidget(left_widget)
            grid_layout.addWidget(right_widget)
            section_layout.addWidget(details_grid)

            if details.get("paths"):
                section_layout.addWidget(self.create_paths_section(details["paths"]))
            
            details_main_layout.addWidget(section)
            self.animate_widget_entry(section, delay=100 * idx, distance=30)
        
        details_main_layout.addStretch()
        self.details_layout.addWidget(details_container)

    def create_detail_item(self, icon, content):
        """Create a responsive detail item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)
        
        icon_label = QLabel(icon)
        icon_label.setFixedWidth(25)
        icon_label.setStyleSheet("font-size: 12px;")
        
        content_label = QLabel(content)
        content_label.setStyleSheet("font-size: 12px; color: #e5e7eb;")
        content_label.setWordWrap(True)
        content_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        layout.addWidget(icon_label)
        layout.addWidget(content_label, 1)
        
        return widget

    def create_pros_cons_section(self, title, items, color):
        """Create responsive pros/cons section"""
        widget = QFrame()
        widget.setStyleSheet(f"""
            QFrame {{
                background: rgba({color[1:]}, 0.1);
                border: 1px solid {color};
                border-radius: 10px;
                padding: 12px;
            }}
        """)
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")
        layout.addWidget(title_label)
        
        for item in items:
            item_label = QLabel(f"• {item}")
            item_label.setStyleSheet("font-size: 11px; color: #e5e7eb; padding: 2px 0;")
            item_label.setWordWrap(True)
            layout.addWidget(item_label)
        
        return widget

    def create_paths_section(self, paths):
        widget = QFrame()
        widget.setObjectName("pathsFrame")
        widget.setStyleSheet("""
            QFrame#pathsFrame {
                background-color: #2a2a2a;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 18px;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        title = QLabel("🧭 Career Path Options")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #fdfdfd;")
        layout.addWidget(title)

        for path in paths:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                background-color: #1f1f1f;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 12px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)

            name = QLabel(path.get("title", "Path"))
            name.setStyleSheet("font-size: 13px; font-weight: bold; color: #f0f0f0;")
            name.setWordWrap(True)
            card_layout.addWidget(name)

            desc = QLabel(path.get("description", ""))
            desc.setStyleSheet("font-size: 12px; color: #d4d4d4;")
            desc.setWordWrap(True)
            card_layout.addWidget(desc)

            layout.addWidget(card)

        return widget

    def get_focus_exam_step(self):
        """Return the mandatory exam step based on science focus selection."""
        if not hasattr(self, "stream_cb") or not self.stream_cb:
            return None
        if self.stream_cb.currentText() != "Science":
            return None

        focus = self.get_science_focus()
        if focus == "Medical":
            return "Mandatory milestone: Crack NEET-UG to secure admission into top medical programs."
        if focus == "Non-Medical":
            return "Mandatory milestone: Clear JEE Main/Advanced or equivalent engineering entrance (CET/BITSAT/VITEEE)."
        return None

    def get_enhanced_roadmap_steps(self, base_steps):
        """Enhance roadmap steps with focus-specific entrance exams."""
        steps = list(base_steps) if base_steps else []
        exam_step = self.get_focus_exam_step()
        if exam_step:
            keyword = "neet" if "NEET" in exam_step.upper() else "jee"
            has_keyword = any(
                isinstance(step, str) and keyword in step.lower()
                for step in steps
            )
            if not has_keyword:
                insert_index = 1 if steps else 0
                steps.insert(insert_index, exam_step)
        return steps

    def has_roadmap_content(self, details):
        """Determine if a career has roadmap info or injected exam milestones."""
        base_steps = details.get("roadmap") or []
        steps = self.get_enhanced_roadmap_steps(base_steps)
        return bool(steps or details.get("sub_specialty_steps"))

    def _register_animation(self, animation, persistent=False):
        """Keep track of running animations to prevent garbage collection."""
        if not hasattr(self, "_active_animations"):
            self._active_animations = []
        self._active_animations.append(animation)
        if not persistent:
            animation.finished.connect(lambda a=animation: self._cleanup_animation(a))

    def _cleanup_animation(self, animation):
        """Remove finished animations from the keep-alive registry."""
        if hasattr(self, "_active_animations") and animation in self._active_animations:
            self._active_animations.remove(animation)

    def animate_widget_entry(self, widget, delay=0, duration=600, direction="up", distance=30):
        """Apply a subtle fade animation when a widget appears."""
        if not widget:
            return

        def start_animation():
            try:
                existing_effect = widget.graphicsEffect()
            except RuntimeError:
                return

            # Avoid overriding existing permanent effects (e.g., glows)
            if existing_effect:
                return

            opacity_effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(opacity_effect)
            opacity_effect.setOpacity(0.0)

            opacity_anim = QPropertyAnimation(opacity_effect, b"opacity", widget)
            opacity_anim.setDuration(duration)
            opacity_anim.setStartValue(0.0)
            opacity_anim.setEndValue(1.0)
            opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

            def finalize():
                try:
                    if widget.graphicsEffect() == opacity_effect:
                        widget.setGraphicsEffect(None)
                except RuntimeError:
                    pass

            opacity_anim.finished.connect(finalize)
            self._register_animation(opacity_anim)
            opacity_anim.start()

        if delay:
            QTimer.singleShot(delay, start_animation)
        else:
            start_animation()

    def apply_button_glow(self, button, color="#60a5fa"):
        """Add a breathing glow animation to highlight primary CTAs."""
        if not button:
            return

        glow_effect = QGraphicsDropShadowEffect(button)
        glow_effect.setColor(QColor(color))
        glow_effect.setBlurRadius(20)
        glow_effect.setOffset(0, 0)
        button.setGraphicsEffect(glow_effect)

        pulse = QPropertyAnimation(glow_effect, b"blurRadius", button)
        pulse.setDuration(1500)
        pulse.setStartValue(15)
        pulse.setEndValue(35)
        pulse.setEasingCurve(QEasingCurve.InOutSine)
        pulse.setLoopCount(-1)

        self._register_animation(pulse, persistent=True)
        pulse.start()
        try:
            button.destroyed.connect(lambda *_: self._cleanup_animation(pulse))
        except AttributeError:
            pass

    def transition_to_page(self, target_widget, duration=450):
        """Smoothly transition between stacked pages with a premium fade."""
        if not target_widget or not hasattr(self, "stack"):
            return

        current_widget = self.stack.currentWidget()
        if current_widget == target_widget:
            return

        if not current_widget:
            self.stack.setCurrentWidget(target_widget)
            return

        current_effect = QGraphicsOpacityEffect(current_widget)
        current_effect.setOpacity(1.0)
        current_widget.setGraphicsEffect(current_effect)

        fade_out = QPropertyAnimation(current_effect, b"opacity", current_widget)
        fade_out.setDuration(duration)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InOutQuad)

        def start_fade_in():
            self.stack.setCurrentWidget(target_widget)
            target_effect = QGraphicsOpacityEffect(target_widget)
            target_effect.setOpacity(0.0)
            target_widget.setGraphicsEffect(target_effect)

            fade_in = QPropertyAnimation(target_effect, b"opacity", target_widget)
            fade_in.setDuration(duration)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.setEasingCurve(QEasingCurve.InOutQuad)

            def cleanup_target():
                target_widget.setGraphicsEffect(None)

            fade_in.finished.connect(cleanup_target)
            self._register_animation(fade_in)
            fade_in.start()

        def cleanup_current():
            current_widget.setGraphicsEffect(None)
            start_fade_in()

        fade_out.finished.connect(cleanup_current)
        self._register_animation(fade_out)
        fade_out.start()

    def create_roadmap_section(self, details):
        base_steps = details.get("roadmap") or []
        steps = self.get_enhanced_roadmap_steps(base_steps)
        specialty_steps = details.get("sub_specialty_steps")
        if not steps and not specialty_steps:
            return None

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setSpacing(15)

        if steps:
            roadmap_frame = QFrame()
            roadmap_frame.setObjectName("roadmapFrame")
            roadmap_frame.setStyleSheet("""
                QFrame#roadmapFrame {
                    background-color: #0d1526;
                    border: 1px solid #1d2a3f;
                    border-radius: 12px;
                    padding: 15px;
                }
            """)
            roadmap_layout = QVBoxLayout(roadmap_frame)
            roadmap_layout.setSpacing(10)

            title = QLabel("🛣️ Step-by-Step Roadmap")
            title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ade80;")
            roadmap_layout.addWidget(title)

            for idx, step in enumerate(steps, 1):
                lbl = QLabel(f"{idx}. {step}")
                lbl.setStyleSheet("font-size: 12px; color: #e2e8f0;")
                lbl.setWordWrap(True)
                roadmap_layout.addWidget(lbl)

            wrapper_layout.addWidget(roadmap_frame)

        if specialty_steps:
            specialty_frame = QFrame()
            specialty_frame.setObjectName("specialtyFrame")
            specialty_frame.setStyleSheet("""
                QFrame#specialtyFrame {
                    background-color: #101a2f;
                    border: 1px solid #243453;
                    border-radius: 12px;
                    padding: 15px;
                }
            """)
            specialty_layout = QVBoxLayout(specialty_frame)
            specialty_layout.setSpacing(12)

            title = QLabel("🎓 Advanced Pathways")
            title.setStyleSheet("font-size: 14px; font-weight: bold; color: #93c5fd;")
            specialty_layout.addWidget(title)

            for name, steps_list in specialty_steps.items():
                card = QFrame()
                card.setStyleSheet("""
                    QFrame {
                        background-color: #111f36;
                        border-radius: 10px;
                        border: 1px solid #1e293b;
                        padding: 10px;
                    }
                """)
                card_layout = QVBoxLayout(card)
                card_layout.setSpacing(6)

                subtitle = QLabel(name)
                subtitle.setStyleSheet("font-size: 13px; font-weight: bold; color: #fcd34d;")
                card_layout.addWidget(subtitle)

                for idx, step in enumerate(steps_list, 1):
                    lbl = QLabel(f"{idx}. {step}")
                    lbl.setStyleSheet("font-size: 12px; color: #cbd5f5;")
                    lbl.setWordWrap(True)
                    card_layout.addWidget(lbl)

                specialty_layout.addWidget(card)

            wrapper_layout.addWidget(specialty_frame)

        return wrapper

    def show_roadmap_dialog(self, job_name, details):
        """Show roadmap in a dialog window"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"🗺️ Roadmap - {job_name}")
        dialog.setMinimumSize(700, 600)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #0a0e1a;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel(f"🛣️ Career Roadmap: {job_name}")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #fbbf24; padding: 10px;")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        # Scroll area for roadmap content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        steps = self.get_enhanced_roadmap_steps(details.get("roadmap") or [])
        specialty_steps = details.get("sub_specialty_steps")
        
        if steps:
            roadmap_frame = QFrame()
            roadmap_frame.setObjectName("roadmapFrame")
            roadmap_frame.setStyleSheet("""
                QFrame#roadmapFrame {
                    background-color: #0d1526;
                    border: 1px solid #1d2a3f;
                    border-radius: 12px;
                    padding: 20px;
                }
            """)
            roadmap_layout = QVBoxLayout(roadmap_frame)
            roadmap_layout.setSpacing(12)

            title_label = QLabel("📋 Step-by-Step Roadmap")
            title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ade80; margin-bottom: 10px;")
            roadmap_layout.addWidget(title_label)

            for idx, step in enumerate(steps, 1):
                step_widget = QWidget()
                step_layout = QHBoxLayout(step_widget)
                step_layout.setSpacing(10)
                
                number_label = QLabel(f"{idx}.")
                number_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #60a5fa; min-width: 30px;")
                step_layout.addWidget(number_label)
                
                step_label = QLabel(step)
                step_label.setStyleSheet("font-size: 13px; color: #e2e8f0;")
                step_label.setWordWrap(True)
                step_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                step_layout.addWidget(step_label)
                
                roadmap_layout.addWidget(step_widget)

            content_layout.addWidget(roadmap_frame)
            self.animate_widget_entry(roadmap_frame, delay=50, distance=20)

        if specialty_steps:
            specialty_frame = QFrame()
            specialty_frame.setObjectName("specialtyFrame")
            specialty_frame.setStyleSheet("""
                QFrame#specialtyFrame {
                    background-color: #101a2f;
                    border: 1px solid #243453;
                    border-radius: 12px;
                    padding: 20px;
                }
            """)
            specialty_layout = QVBoxLayout(specialty_frame)
            specialty_layout.setSpacing(15)

            title_label = QLabel("🎓 Advanced Pathways & Specializations")
            title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #93c5fd; margin-bottom: 10px;")
            specialty_layout.addWidget(title_label)

            for name, steps_list in specialty_steps.items():
                card = QFrame()
                card.setStyleSheet("""
                    QFrame {
                        background-color: #111f36;
                        border-radius: 10px;
                        border: 1px solid #1e293b;
                        padding: 15px;
                    }
                """)
                card_layout = QVBoxLayout(card)
                card_layout.setSpacing(8)

                subtitle = QLabel(name)
                subtitle.setStyleSheet("font-size: 14px; font-weight: bold; color: #fcd34d; margin-bottom: 5px;")
                card_layout.addWidget(subtitle)

                for idx, step in enumerate(steps_list, 1):
                    step_widget = QWidget()
                    step_layout = QHBoxLayout(step_widget)
                    step_layout.setSpacing(10)
                    
                    number_label = QLabel(f"{idx}.")
                    number_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #60a5fa; min-width: 25px;")
                    step_layout.addWidget(number_label)
                    
                    step_label = QLabel(step)
                    step_label.setStyleSheet("font-size: 12px; color: #cbd5f5;")
                    step_label.setWordWrap(True)
                    step_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    step_layout.addWidget(step_label)
                    
                    card_layout.addWidget(step_widget)

                specialty_layout.addWidget(card)

            content_layout.addWidget(specialty_frame)
            self.animate_widget_entry(specialty_frame, delay=80, distance=20)
        
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def show_colleges_dialog(self, job_name):
        """Show curated college information for the given career."""
        colleges = self.get_college_info(job_name)
        if not colleges:
            QMessageBox.information(self, "Colleges Information", "College recommendations coming soon.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"🏫 Colleges Information - {job_name}")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1f1f1f;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Top Colleges & Exams for {job_name}")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5f5f5;")
        title.setWordWrap(True)
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        content_layout.setContentsMargins(5, 5, 5, 5)

        for idx, college in enumerate(colleges, 1):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #2a2a2a;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                    padding: 14px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)

            name = QLabel(f"{idx}. {college.get('name', 'College')}")
            name.setStyleSheet("font-size: 15px; font-weight: bold; color: #f5f5f5;")
            name.setWordWrap(True)
            card_layout.addWidget(name)

            exam = QLabel(f"Entrance Exam: {college.get('exam', 'Varies')}")
            exam.setStyleSheet("font-size: 13px; color: #d4d4d4;")
            exam.setWordWrap(True)
            card_layout.addWidget(exam)

            highlights = QLabel(college.get('highlights', 'Known for academic excellence.'))
            highlights.setStyleSheet("font-size: 12px; color: #c9c9c9;")
            highlights.setWordWrap(True)
            card_layout.addWidget(highlights)

            content_layout.addWidget(card)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c2c2c;
                color: #f5f5f5;
                border: none;
                border-radius: 8px;
                padding: 10px 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dialog.exec()

    def display_analytics(self, recommendations):
        """Display responsive analytics and charts"""
        analytics_container = QWidget()
        analytics_main_layout = QVBoxLayout(analytics_container)
        analytics_main_layout.setSpacing(15)
        analytics_main_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel("📊 Career Match Analytics")
        header.setStyleSheet("font-size: clamp(18px, 3vw, 20px); font-weight: bold; color: #60a5fa; text-align: center;")
        analytics_main_layout.addWidget(header)

        # Create responsive chart
        names = [job for job, _ in recommendations]
        scores = [score * 100 for _, score in recommendations]
        
        # Adjust figure size based on available space
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#2c2c2c')
        ax.set_facecolor('#1a1a1a')
        
        colors = ['#60a5fa', '#34d399', '#fbbf24', '#a78bfa']
        bars = ax.bar(names, scores, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_ylabel('Match Score (%)', color='white', fontsize=11)
        ax.set_title('Career Recommendation Scores', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10, color='white')
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        analytics_main_layout.addWidget(canvas)
        analytics_main_layout.addStretch()
        
        self.analytics_layout.addWidget(analytics_container)
        self.animate_widget_entry(analytics_container, delay=120, distance=30)

    def display_resume_builder(self):
        """Display resume builder section"""
        resume_container = QWidget()
        resume_main_layout = QVBoxLayout(resume_container)
        resume_main_layout.setSpacing(20)
        resume_main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("📄 Automatic Resume Builder")
        header.setStyleSheet("""
            font-size: clamp(20px, 3vw, 24px); 
            font-weight: bold; 
            color: #60a5fa; 
            text-align: center;
            padding: 10px;
        """)
        resume_main_layout.addWidget(header)

        # Description
        desc = QLabel("Generate a professional resume automatically based on your career preferences and personal information. The resume will be tailored to your recommended career path.")
        desc.setStyleSheet("font-size: 14px; color: #cbd5e1; line-height: 1.5; text-align: center;")
        desc.setWordWrap(True)
        resume_main_layout.addWidget(desc)

        # Features section
        features_frame = QFrame()
        features_frame.setObjectName("glassFrame")
        features_layout = QVBoxLayout(features_frame)
        features_layout.setSpacing(15)

        features_title = QLabel("✨ Resume Features")
        features_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fbbf24;")
        features_layout.addWidget(features_title)

        features = [
            "✅ Professional ATS-friendly format",
            "✅ Personalized career objective",
            "✅ Skills tailored to your interests",
            "✅ Relevant project suggestions", 
            "✅ Education section optimized for your career",
            "✅ Achievements and certifications section",
            "✅ Download as PDF format"
        ]

        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("font-size: 14px; color: #e5e7eb; padding: 5px 0;")
            features_layout.addWidget(feature_label)

        resume_main_layout.addWidget(features_frame)

        # Generate Resume Button
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        
        generate_btn = QPushButton("🔄 Generate My Resume PDF")
        generate_btn.setMinimumHeight(50)
        generate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #34d399);
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #10b981);
            }
        """)
        generate_btn.clicked.connect(self.generate_resume_pdf)
        self.apply_button_glow(generate_btn, color="#34d399")
        
        button_layout.addStretch()
        button_layout.addWidget(generate_btn)
        button_layout.addStretch()
        
        resume_main_layout.addWidget(button_container)
        resume_main_layout.addStretch()

        self.resume_layout.addWidget(resume_container)
        self.animate_widget_entry(resume_container, delay=150, distance=35)

    def generate_resume_pdf(self):
        """Generate and download resume PDF"""
        try:
            # Get user data from inputs
            user_data = {
                'stream': self.stream_cb.currentText(),
                'field': self.field_cb.currentText(),
                'role': self.role_cb.currentText(),
                'hobby': self.inputs['hobby'].currentText(),
                'free_time': self.inputs['free_time'].currentText(),
                'interested_subject': self.inputs['interested_subject'].currentText(),
                'interests': self.free_text.toPlainText().strip()
            }

            # Show personal info dialog
            dialog = PersonalInfoDialog(self)
            if dialog.exec() == QDialog.Accepted:
                personal_info = dialog.get_personal_info()
                
                # Merge personal info with user data
                user_data.update(personal_info)
                
                # Generate filename
                name = personal_info.get('name', 'Resume').replace(' ', '_')
                filename = f"{name}_Career_Resume.pdf"
                
                # Create resume
                resume_path = self.resume_builder.create_resume(
                    user_data, 
                    self.current_recommendations, 
                    filename
                )
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Resume Generated Successfully", 
                    f"Your professional resume has been generated!\n\n"
                    f"File saved as: {filename}\n\n"
                    f"You can find it in the current directory.",
                    QMessageBox.Ok
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Resume Generation Failed", 
                f"Could not generate resume:\n{str(e)}",
                QMessageBox.Ok
            )


# ---------- RUN APP ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CareerApp()
    window.show()
    sys.exit(app.exec())