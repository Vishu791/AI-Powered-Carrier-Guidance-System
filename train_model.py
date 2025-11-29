# train_model.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def build_dataset():
    """
    Returns:
        X: list of input texts (stream + interests + skills)
        y: list of labels (career/job role)
    The more examples you add here, the better the model.
    """

    X = [
        # ---------- COMPUTER SCIENCE / IT ----------
        "computer science coding programming software apps websites backend frontend",
        "cse love coding want to develop apps and websites full stack",
        "computer science interested in java python c++ software development",
        "cse like solving problems data structures algorithms software engineer",
        "it want to become app developer android ios mobile applications",
        "like unity unreal games development c# c plus plus game developer",
        "interested in web development html css javascript frontend developer",
        "interested in backend development apis databases nodejs backend developer",
        "cse cloud computing devops docker kubernetes aws azure",
        "like managing servers deployment ci cd pipelines devops engineer",
        "network security hacking cyber security analyst",
        "ethical hacking bug bounty penetration testing cyber security",
        "machine learning artificial intelligence data science python statistics",
        "love working with data analysis visualization data analyst",
        "ai ml deep learning neural networks want ai engineer career",

        # ---------- VLSI / EMBEDDED / HARDWARE ----------
        "electronics vlsi chip design verilog digital design semiconductor",
        "ece interested in embedded systems microcontrollers iot",
        "electrical and electronics want to design circuits hardware engineer",

        # ---------- MECHANICAL ----------
        "mechanical engineering like machines engines automotive",
        "mechanical cad cam designing parts solidworks autocad",
        "mechanical interested in hvac heating ventilation air conditioning",

        # ---------- CIVIL ----------
        "civil engineering like construction buildings roads bridges",
        "civil enjoy planning layout maps structural design",

        # ---------- ELECTRICAL ----------
        "electrical engineering interested in power systems power plants grids",
        "eee like working with electronics control systems embedded",

        # ---------- CHEMICAL ----------
        "chemical engineering interested in process plants oil gas refinery",
        "chemical like pharma medicines production research",

        # ---------- ARTS / HUMANITIES ----------
        "arts like writing stories blogs content creator",
        "journalism mass communication news reporting anchor media",
        "interested in psychology helping people counsellor",
        "love social work ngo helping society social worker",
        "creative graphic design logo posters illustrator photoshop designer",
        "interested in ui ux designing apps websites user experience",

        # ---------- PURE SCIENCE ----------
        "bsc physics enjoy research space astronomy scientist",
        "bsc chemistry lab work formulations chemist",
        "bsc biology microbiology genetics lab biotech researcher",
        "good at maths statistics data want analytic role",

        # ---------- COMMERCE ----------
        "commerce like accounts balance sheet ca chartered accountant",
        "bcom interested in taxation gst accounting finance",
        "like stock market investment banking finance analyst",
        "mba marketing like sales branding business development",
        "interested in hr human resources recruitment training people management",
        "economics like studying market inflation economic analyst",
        "want to start my own business startup entrepreneurship",
        "family business background want to expand startup founder entrepreneur",
    ]

    y = [
        # ---------- COMPUTER SCIENCE / IT ----------
        "Software Engineer",
        "Full Stack Developer",
        "Software Engineer",
        "Software Engineer",
        "Mobile App Developer",
        "Game Developer",
        "Frontend Developer",
        "Backend Developer",
        "Cloud / DevOps Engineer",
        "DevOps Engineer",
        "Cybersecurity Analyst",
        "Ethical Hacker",
        "Data Scientist / ML Engineer",
        "Data Analyst",
        "AI Engineer",

        # ---------- VLSI / EMBEDDED / HARDWARE ----------
        "VLSI / Chip Design Engineer",
        "Embedded Systems Engineer",
        "Hardware Engineer",

        # ---------- MECHANICAL ----------
        "Automotive / Mechanical Design Engineer",
        "Mechanical Design Engineer",
        "HVAC Engineer",

        # ---------- CIVIL ----------
        "Civil Site Engineer",
        "Structural / Planning Engineer",

        # ---------- ELECTRICAL ----------
        "Power Systems Engineer",
        "Electronics / Control Systems Engineer",

        # ---------- CHEMICAL ----------
        "Process Engineer",
        "Pharmaceutical / Chemical Industry Role",

        # ---------- ARTS / HUMANITIES ----------
        "Content Writer / Blogger",
        "Journalist / Media Professional",
        "Psychologist / Counselor",
        "Social Worker / NGO Professional",
        "Graphic Designer",
        "UI/UX Designer",

        # ---------- PURE SCIENCE ----------
        "Physics Researcher / Scientist",
        "Chemist / Lab Scientist",
        "Biotech / Microbiology Researcher",
        "Data Scientist / Statistician",

        # ---------- COMMERCE ----------
        "Chartered Accountant / Accountant",
        "Accountant / Tax Consultant",
        "Finance / Investment Analyst",
        "Marketing / Sales Manager",
        "HR Manager",
        "Economist / Policy Analyst",
        "Entrepreneur / Startup Founder",
        "Entrepreneur / Startup Founder",
    ]

    return X, y


def train_and_save_model():
    X, y = build_dataset()

    # Pipeline: TF-IDF text features + Linear SVM classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    model.fit(X, y)

    # Save the model
    joblib.dump(model, "career_model.pkl")
    print("✅ Saved trained model to career_model.pkl")

    # Optional: extra details for each career label
    career_details = {
        "Software Engineer": {
            "sector": "Computer Science & Engineering",
            "skills": "Programming (Python/Java/C++), DSA, OOP, debugging, Git",
            "description": "Designs and builds software applications like websites, apps, tools.",
            "future_scope": "Very high demand in IT, product-based companies, startups."
        },
        "Full Stack Developer": {
            "sector": "Computer Science & Engineering",
            "skills": "HTML, CSS, JS, React, Node.js/Java/Python, databases",
            "description": "Works on both frontend (UI) and backend (server, database).",
            "future_scope": "Always in demand for startups and web-based companies."
        },
        "Mobile App Developer": {
            "sector": "Computer Science & Engineering",
            "skills": "Android (Kotlin/Java), iOS (Swift), Flutter/React Native",
            "description": "Builds apps for Android/iOS phones.",
            "future_scope": "High demand due to mobile-first world."
        },
        "Game Developer": {
            "sector": "Computer Science & Engineering",
            "skills": "Unity/Unreal, C#, C++, graphics programming",
            "description": "Creates 2D/3D games and interactive experiences.",
            "future_scope": "Growing gaming and AR/VR industry."
        },
        "Frontend Developer": {
            "sector": "Computer Science & Engineering",
            "skills": "HTML, CSS, JavaScript, React/Angular",
            "description": "Builds user interfaces for websites and web apps.",
            "future_scope": "High demand in web and SaaS companies."
        },
        "Backend Developer": {
            "sector": "Computer Science & Engineering",
            "skills": "Node.js/Java/Python, APIs, SQL/NoSQL, security",
            "description": "Handles server-side logic and databases.",
            "future_scope": "Core backend roles are always required."
        },
        "Cloud / DevOps Engineer": {
            "sector": "Computer Science & Engineering",
            "skills": "Linux, AWS/Azure/GCP, Docker, Kubernetes, CI/CD",
            "description": "Manages cloud infrastructure and deployment pipelines.",
            "future_scope": "Huge scope as companies shift to cloud."
        },
        "DevOps Engineer": {
            "sector": "Computer Science & Engineering",
            "skills": "Automation, CI/CD, scripting, cloud tools",
            "description": "Bridges development and operations for faster delivery.",
            "future_scope": "Very strong demand in modern software teams."
        },
        "Cybersecurity Analyst": {
            "sector": "Computer Science & Engineering",
            "skills": "Networking, security tools, monitoring, SIEM",
            "description": "Monitors and protects systems from cyber attacks.",
            "future_scope": "Shortage of skilled professionals worldwide."
        },
        "Ethical Hacker": {
            "sector": "Computer Science & Engineering",
            "skills": "Pen-testing, exploit analysis, Kali Linux, bug bounty",
            "description": "Finds security vulnerabilities legally to improve security.",
            "future_scope": "Growing need in all sectors."
        },
        "Data Scientist / ML Engineer": {
            "sector": "Computer Science & Science (Maths)",
            "skills": "Python, ML, statistics, Pandas, visualization",
            "description": "Extracts insights and builds predictive models from data.",
            "future_scope": "One of the top trending careers."
        },
        "AI Engineer": {
            "sector": "Computer Science & Engineering",
            "skills": "Deep learning, neural networks, Python, ML frameworks",
            "description": "Builds AI systems like recommendation engines and chatbots.",
            "future_scope": "Extremely high growth area."
        },
        "Data Analyst": {
            "sector": "Computer Science & Commerce",
            "skills": "SQL, Excel, BI tools, basic statistics",
            "description": "Analyzes data and creates reports/dashboards.",
            "future_scope": "Needed in almost every industry."
        },
        "VLSI / Chip Design Engineer": {
            "sector": "Electronics / VLSI",
            "skills": "Digital design, Verilog/VHDL, CMOS, EDA tools",
            "description": "Designs integrated circuits and chips.",
            "future_scope": "Good scope due to semiconductor push in India."
        },
        "Embedded Systems Engineer": {
            "sector": "Electronics / Embedded",
            "skills": "C, microcontrollers, RTOS, sensors, IoT",
            "description": "Programs hardware devices like IoT, controllers.",
            "future_scope": "High scope in automotive, IoT, robotics."
        },
        "Hardware Engineer": {
            "sector": "Electronics / Hardware",
            "skills": "PCB design, circuits, testing, debugging",
            "description": "Works on physical components of computers/electronics.",
            "future_scope": "Stable demand in manufacturing and devices."
        },
        "Automotive / Mechanical Design Engineer": {
            "sector": "Mechanical Engineering",
            "skills": "Mechanics, CAD tools, materials",
            "description": "Designs mechanical systems, especially vehicles.",
            "future_scope": "Scope in EV and automotive industries."
        },
        "Mechanical Design Engineer": {
            "sector": "Mechanical Engineering",
            "skills": "SolidWorks, AutoCAD, machine design",
            "description": "Designs mechanical parts and assemblies.",
            "future_scope": "Needed in manufacturing and product design."
        },
        "HVAC Engineer": {
            "sector": "Mechanical Engineering",
            "skills": "Thermodynamics, HVAC design, load calculations",
            "description": "Designs heating, ventilation, and air conditioning systems.",
            "future_scope": "Good scope in construction and infrastructure."
        },
        "Civil Site Engineer": {
            "sector": "Civil Engineering",
            "skills": "Construction management, surveying, site supervision",
            "description": "Handles on-site construction work and coordination.",
            "future_scope": "Consistent demand in infrastructure projects."
        },
        "Structural / Planning Engineer": {
            "sector": "Civil Engineering",
            "skills": "Structural analysis, AutoCAD, design codes",
            "description": "Designs safe structures and prepares plans.",
            "future_scope": "Scope in urban development projects."
        },
        "Power Systems Engineer": {
            "sector": "Electrical Engineering",
            "skills": "Power systems, grids, transformers",
            "description": "Works with power generation and distribution.",
            "future_scope": "Important for renewable energy integration."
        },
        "Electronics / Control Systems Engineer": {
            "sector": "Electrical / Electronics",
            "skills": "Control theory, electronics, PLCs",
            "description": "Designs control systems for machines and processes.",
            "future_scope": "Used heavily in automation and robotics."
        },
        "Process Engineer": {
            "sector": "Chemical Engineering",
            "skills": "Process design, P&ID, thermodynamics",
            "description": "Optimizes chemical processes in plants.",
            "future_scope": "Scope in oil, gas, chemical industries."
        },
        "Pharmaceutical / Chemical Industry Role": {
            "sector": "Chemical / Pharma",
            "skills": "Organic chemistry, lab work, QA/QC",
            "description": "Works in drug manufacturing and quality control.",
            "future_scope": "Strong due to pharma growth."
        },
        "Content Writer / Blogger": {
            "sector": "Arts / Media",
            "skills": "Writing, grammar, storytelling",
            "description": "Creates written content for blogs, ads, scripts.",
            "future_scope": "High demand in digital marketing."
        },
        "Journalist / Media Professional": {
            "sector": "Arts / Media",
            "skills": "Reporting, communication, research",
            "description": "Covers news, events, creates media content.",
            "future_scope": "Evolving with digital media platforms."
        },
        "Psychologist / Counselor": {
            "sector": "Arts / Psychology",
            "skills": "Counseling, empathy, assessment tools",
            "description": "Helps people with mental and emotional issues.",
            "future_scope": "Growing awareness of mental health."
        },
        "Social Worker / NGO Professional": {
            "sector": "Social Sciences",
            "skills": "Community work, communication, empathy",
            "description": "Works for social causes and community development.",
            "future_scope": "Impactful work with NGOs and government schemes."
        },
        "Graphic Designer": {
            "sector": "Design / Creative",
            "skills": "Photoshop, Illustrator, creativity",
            "description": "Designs graphics, posters, social media creatives.",
            "future_scope": "Very popular in branding, marketing."
        },
        "UI/UX Designer": {
            "sector": "Design / Tech",
            "skills": "Figma, wireframing, prototyping, user research",
            "description": "Designs user-friendly app and website interfaces.",
            "future_scope": "High demand in product and IT companies."
        },
        "Physics Researcher / Scientist": {
            "sector": "Science",
            "skills": "Physics, maths, experimentation",
            "description": "Works on advanced physics problems and experiments.",
            "future_scope": "Opportunities in research labs, space agencies."
        },
        "Chemist / Lab Scientist": {
            "sector": "Science",
            "skills": "Lab techniques, chemistry, analysis",
            "description": "Works in labs for testing, analysis, research.",
            "future_scope": "Scope in pharma, food, environment labs."
        },
        "Biotech / Microbiology Researcher": {
            "sector": "Science / Biology",
            "skills": "Microbiology, genetics, lab work",
            "description": "Works on vaccines, genetics, microorganisms.",
            "future_scope": "High in biotech and healthcare."
        },
        "Data Scientist / Statistician": {
            "sector": "Science / Data",
            "skills": "Statistics, programming, ML",
            "description": "Uses maths and data to derive insights.",
            "future_scope": "Very strong in tech and finance."
        },
        "Chartered Accountant / Accountant": {
            "sector": "Commerce / Finance",
            "skills": "Accounting, auditing, taxation",
            "description": "Manages company accounts, audits, financial reports.",
            "future_scope": "Evergreen profession in finance."
        },
        "Accountant / Tax Consultant": {
            "sector": "Commerce / Finance",
            "skills": "Tally, GST, income tax",
            "description": "Handles tax filings and bookkeeping.",
            "future_scope": "Required by all businesses."
        },
        "Finance / Investment Analyst": {
            "sector": "Commerce / Finance",
            "skills": "Financial modelling, markets, Excel",
            "description": "Analyzes stocks, companies, investments.",
            "future_scope": "High-paying roles in banks and firms."
        },
        "Marketing / Sales Manager": {
            "sector": "Business / Management",
            "skills": "Communication, persuasion, branding",
            "description": "Promotes products, handles sales and marketing strategy.",
            "future_scope": "Needed in all industries."
        },
        "HR Manager": {
            "sector": "Business / HR",
            "skills": "Recruitment, people management, policies",
            "description": "Manages hiring, training, employee relations.",
            "future_scope": "Important function in all companies."
        },
        "Economist / Policy Analyst": {
            "sector": "Economics",
            "skills": "Economic theory, statistics, modelling",
            "description": "Studies economy, policies, market behaviour.",
            "future_scope": "Scope in government, research, banks."
        },
        "Entrepreneur / Startup Founder": {
            "sector": "Business / Entrepreneurship",
            "skills": "Leadership, risk-taking, management",
            "description": "Starts and runs own business or startup.",
            "future_scope": "High risk, high reward; supported by startup ecosystem."
        },
    }

    joblib.dump(career_details, "career_details.pkl")
    print("✅ Saved career details to career_details.pkl")


if __name__ == "__main__":
    train_and_save_model()