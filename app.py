from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
from fpdf import FPDF
import os

app = Flask(__name__)

# === Load Model Components ===
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')  # StandardScaler
    feature_columns = joblib.load('columns.pkl')  # List of feature columns
    le = joblib.load('le.pkl')  # LabelEncoder
except Exception as e:
    print(f"Error loading model components: {e}")

@app.route('/')
def home():
    return render_template('newui.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = request.form.to_dict()
        processed_input = {}

        # Convert input types
        for key, value in form_values.items():
            if value.lower() == 'yes':
                processed_input[key] = 1.0
            elif value.lower() == 'no':
                processed_input[key] = 0.0
            else:
                try:
                    processed_input[key] = float(value)
                except ValueError:
                    processed_input[key] = value

        input_df = pd.DataFrame([processed_input])

        # === Domain Feature Engineering ===
        input_df['Knowledge Engineering'] = (input_df['percentage in Algorithms'] + input_df['Percentage in Mathematics']) / 2
        input_df['System Engineering'] = (
            input_df['Acedamic percentage in Operating Systems'] +
            input_df['Percentage in Computer Architecture'] +
            input_df['Percentage in Electronics Subjects']) / 3
        input_df['Networks and Security'] = (
            input_df['Percentage in Computer Networks'] +
            input_df['Percentage in Communication skills']) / 2
        input_df['Software Development'] = (
            input_df['Percentage in Programming Concepts'] +
            input_df['Percentage in Software Engineering']) / 2
        input_df['Professional Development'] = (
            input_df['Percentage in Communication skills'] +
            input_df['Percentage in Mathematics']) / 2

        input_df.drop([  # Dropping used columns
            'percentage in Algorithms', 'Percentage in Mathematics',
            'Acedamic percentage in Operating Systems',
            'Percentage in Computer Architecture',
            'Percentage in Electronics Subjects',
            'Percentage in Computer Networks',
            'Percentage in Communication skills',
            'Percentage in Programming Concepts',
            'Percentage in Software Engineering'
        ], axis=1, inplace=True, errors='ignore')

        # One-hot encode and reindex
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        decoded_prediction = le.inverse_transform(prediction)

        # Store prediction for PDF download
        with open('last_prediction.txt', 'w') as f:
            f.write(decoded_prediction[0])

        # === Generate PDF with Career Role Info ===
        role_info = jobRoleInfo.get(prediction[0], {})
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, txt=f"Predicted Role: {decoded_prediction[0]}", ln=True, align='C')

        pdf.ln(10)  # Line break
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, f"Description: {role_info.get('Description', 'N/A')}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Skills: {role_info.get('Skills', 'N/A')}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Potential Companies: {role_info.get('PotentialCompanies', 'N/A')}")

        pdf_output_path = 'career_prediction.pdf'
        pdf.output(pdf_output_path)

        # Return the generated PDF file for download
        return send_file(pdf_output_path, as_attachment=True, download_name='career_prediction.pdf')

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('newui.html', prediction_text=f' Error: {str(e)}')
    
jobRoleInfo = {
    1: {
        "Role": "Database Developer",
        "Description": "Focuses on creating and managing databases, ensuring efficiency and accuracy through design, performance optimization, and data migration.",
        "Skills": "SQL, database design (Normalization), DBMS (Oracle, MySQL, PostgreSQL, MS SQL Server), performance tuning, data migration.",
        "PotentialCompanies": "TCS, Infosys, Capgemini, L&T Infotech (LTI), Tech Mahindra, Accenture, JP Morgan Chase, Reliance Industries, ICICI Bank."
    },
    2: {
        "Role": "Portal Administrator",
        "Description": "Manages user accounts, permissions, content updates, and troubleshooting within web portals.",
        "Skills": "CMS (SharePoint, Drupal), user account management, access control, web security basics, troubleshooting.",
        "PotentialCompanies": "HCLTech, Wipro, Cognizant, Capgemini, Godrej Group, educational institutions, government organizations."
    },
    3: {
        "Role": "Systems Security Administrator",
        "Description": "Ensures the security of an organization's IT infrastructure by implementing security protocols and monitoring for potential threats.",
        "Skills": "Firewalls, intrusion detection systems, encryption, security policies, access control, ethical hacking.",
        "PotentialCompanies": "McAfee, Palo Alto Networks, IBM, Cisco, Accenture, Infosys, Wipro."
    },
    4: {
        "Role": "Business Systems Analyst",
        "Description": "Acts as a liaison between business stakeholders and IT teams to gather and document requirements, analyze processes, and propose IT solutions.",
        "Skills": "Business analysis, documentation, requirements gathering, use case creation, communication, and analytical skills.",
        "PotentialCompanies": "TCS, Infosys, Wipro, Deloitte, EY, PwC, Kotak Mahindra Bank, pharmaceutical companies, retail companies."
    },
    5: {
        "Role": "Software Systems Engineer",
        "Description": "Designs, develops, and maintains software systems and applications, ensuring they meet the required specifications and performance standards.",
        "Skills": "Java, Python, C++, software design patterns, debugging, version control (Git), unit testing, SDLC.",
        "PotentialCompanies": "Google, Microsoft, Amazon, Facebook, LinkedIn, Adobe, Salesforce."
    },
    6: {
        "Role": "Business Intelligence Analyst",
        "Description": "Uses data analysis tools to generate insights and create reports for business decision-making.",
        "Skills": "Tableau, Power BI, SAP BusinessObjects, data visualization, report generation, KPIs, and metrics analysis.",
        "PotentialCompanies": "Flipkart, Amazon India, Walmart Labs, Reliance Industries, consulting firms with BI practices, market research firms, retail chains."
    },
    7: {
        "Role": "CRM Technical Developer",
        "Description": "Develops and customizes CRM software solutions to meet business requirements, integrating systems and automating processes.",
        "Skills": "CRM platforms (Salesforce, Microsoft Dynamics), Java, SQL, integration tools, APIs, business analysis.",
        "PotentialCompanies": "Salesforce, Microsoft, Oracle, SAP, consulting firms, large enterprises."
    },
    8: {
        "Role": "Mobile Applications Developer",
        "Description": "Develops applications for smartphones and tablets, ensuring high performance, usability, and design.",
        "Skills": "Swift/Objective-C (iOS), Java/Kotlin (Android), UI/UX design, mobile app architecture, testing, performance optimization.",
        "PotentialCompanies": "Byju's, Swiggy, Zomato, Ola Cabs, Flipkart, Amazon India, app development agencies."
    },
    9: {
        "Role": "UX Designer",
        "Description": "Designs user experiences for websites and applications, focusing on improving usability and customer satisfaction.",
        "Skills": "Wireframing, prototyping, user testing, Sketch, Figma, Adobe XD, design principles, UI development.",
        "PotentialCompanies": "Apple, Google, Amazon, Facebook, startups, design agencies."
    },
    10: {
        "Role": "Quality Assurance Associate",
        "Description": "Ensures the quality of software applications through testing, bug identification, and creating test cases.",
        "Skills": "Manual testing, automation testing (Selenium, JUnit), test cases, bug tracking, agile methodologies.",
        "PotentialCompanies": "Accenture, Capgemini, Cognizant, Wipro, Infosys."
    },
    11: {
        "Role": "Web Developer",
        "Description": "Designs and develops websites and web applications, ensuring functionality, accessibility, and responsiveness.",
        "Skills": "HTML, CSS, JavaScript, React, Angular, Node.js, PHP, MySQL, web frameworks.",
        "PotentialCompanies": "Google, Facebook, LinkedIn, Amazon, startups, digital agencies."
    },
    12: {
        "Role": "Information Security Analyst",
        "Description": "Monitors and protects an organization's IT infrastructure from security breaches and cyberattacks.",
        "Skills": "Network security, encryption, firewalls, ethical hacking, penetration testing, security protocols.",
        "PotentialCompanies": "McAfee, Symantec, Palo Alto Networks, FireEye, Cisco, IBM, Wipro, Infosys."
    },
    13: {
        "Role": "CRM Business Analyst",
        "Description": "Analyzes and defines business requirements for CRM systems, working closely with stakeholders to improve customer relations.",
        "Skills": "CRM tools (Salesforce, Microsoft Dynamics), business analysis, requirements gathering, process mapping, project management.",
        "PotentialCompanies": "Salesforce, Microsoft, Oracle, SAP, consulting firms, large enterprises."
    },
    14: {
        "Role": "Technical Support",
        "Description": "Provides technical assistance to clients, troubleshoots software or hardware issues, and ensures customer satisfaction.",
        "Skills": "Problem-solving, troubleshooting, networking, IT support, communication skills.",
        "PotentialCompanies": "Dell, HP, IBM, Infosys, Wipro, Zoho, Oracle."
    },
    15: {
        "Role": "Project Manager",
        "Description": "Manages IT projects, coordinating teams, timelines, and resources to ensure successful project delivery.",
        "Skills": "Project management, Agile, Scrum, resource allocation, risk management, communication.",
        "PotentialCompanies": "TCS, Infosys, Cognizant, Deloitte, Accenture."
    },
    16: {
        "Role": "Information Technology Manager",
        "Description": "Oversees the IT department and its operations, ensuring technology infrastructure meets business needs.",
        "Skills": "IT management, team leadership, infrastructure management, budgeting, IT strategy, cybersecurity.",
        "PotentialCompanies": "Cisco, IBM, Wipro, Infosys, Accenture."
    },
    17: {
        "Role": "Programmer Analyst",
        "Description": "Develops and maintains software applications based on business requirements, ensuring optimal performance and scalability.",
        "Skills": "Java, C++, Python, SQL, application development, debugging, problem-solving.",
        "PotentialCompanies": "TCS, Wipro, Cognizant, Accenture, Infosys."
    },
    18: {
        "Role": "Design & UX",
        "Description": "Focuses on designing the user experience of digital products, ensuring ease of use and user satisfaction.",
        "Skills": "Wireframing, prototyping, user testing, user interface design, design principles, UX tools.",
        "PotentialCompanies": "Apple, Google, Microsoft, startups, design agencies."
    },
    19: {
        "Role": "Solutions Architect",
        "Description": "Designs and implements technology solutions to meet business needs, focusing on scalability, security, and performance.",
        "Skills": "System architecture, cloud computing, solution design, business analysis, project management.",
        "PotentialCompanies": "Amazon, Microsoft, Google, consulting firms, large enterprises."
    },
    20: {
        "Role": "Systems Analyst",
        "Description": "Analyzes and designs IT systems and software, ensuring they meet business requirements and performance standards.",
        "Skills": "System analysis, programming, problem-solving, SQL, requirements gathering, SDLC.",
        "PotentialCompanies": "IBM, Oracle, TCS, Infosys, Capgemini."
    },
    21: {
        "Role": "Network Security Administrator",
        "Description": "Monitors and manages network security infrastructure, protecting systems from external and internal threats.",
        "Skills": "Firewall configuration, VPN, intrusion detection, network protocols, network security tools.",
        "PotentialCompanies": "Cisco, Palo Alto Networks, Symantec, IBM, government organizations."
    },
    22: {
        "Role": "Data Architect",
        "Description": "Designs and manages data systems, ensuring they meet business requirements and are scalable and secure.",
        "Skills": "Data modeling, database design, SQL, cloud computing, ETL, big data technologies.",
        "PotentialCompanies": "Google, Amazon, Microsoft, data analytics companies, consulting firms."
    },
    23: {
        "Role": "Software Developer",
        "Description": "Develops, tests, and maintains software applications according to user needs and business requirements.",
        "Skills": "Java, Python, C++, software design, debugging, problem-solving, agile methodologies.",
        "PotentialCompanies": "TCS, Infosys, Wipro, Accenture, Capgemini, startups."
    },
    24: {
        "Role": "E-Commerce Analyst",
        "Description": "Analyzes e-commerce trends, user behavior, and sales data to improve online sales and marketing strategies.",
        "Skills": "Google Analytics, SEO, SEM, data analysis, conversion optimization, customer segmentation.",
        "PotentialCompanies": "Amazon, Flipkart, Walmart, e-commerce startups."
    },
    25: {
        "Role": "Technical Services/Help Desk/Tech Support",
        "Description": "Provides technical assistance and support to end-users, troubleshooting and resolving IT-related issues.",
        "Skills": "Technical troubleshooting, customer service, IT support, software and hardware knowledge.",
        "PotentialCompanies": "Dell, HP, IBM, Zoho, Oracle, tech startups."
    },
    26: {
        "Role": "Information Technology Auditor",
        "Description": "Conducts audits of IT systems and processes to ensure compliance with industry standards and regulations.",
        "Skills": "Auditing, risk assessment, compliance, IT controls, cybersecurity, ISO standards.",
        "PotentialCompanies": "PwC, KPMG, Deloitte, EY, consulting firms, government organizations."
    },
    27: {
        "Role": "Database Manager",
        "Description": "Manages database systems and their performance, ensuring they are secure, efficient, and meet business requirements.",
        "Skills": "Database management, SQL, DBMS (Oracle, MySQL), database performance tuning, backup strategies.",
        "PotentialCompanies": "TCS, Infosys, Wipro, Oracle, SAP."
    },
    28: {
        "Role": "Applications Developer",
        "Description": "Designs, develops, and tests software applications that meet user needs and improve business processes.",
        "Skills": "Java, Python, application development, software testing, problem-solving, version control.",
        "PotentialCompanies": "Google, Microsoft, Amazon, Accenture, consulting firms."
    },
    29: {
        "Role": "Database Administrator",
        "Description": "Manages and maintains databases, ensuring security, performance, and availability of data.",
        "Skills": "SQL, Oracle, MySQL, database backup, performance tuning, DBMS management.",
        "PotentialCompanies": "TCS, Infosys, Wipro, Oracle, Microsoft."
    },
    30: {
        "Role": "Network Engineer",
        "Description": "Designs, implements, and maintains network infrastructure, ensuring secure and efficient communication across systems.",
        "Skills": "TCP/IP, VPN, network protocols, routers, switches, firewalls, network troubleshooting.",
        "PotentialCompanies": "Cisco, Juniper Networks, Huawei, IBM, Wipro, Tech Mahindra, TCS."
    },
    31: {
        "Role": "Software Engineer",
        "Description": "Designs, develops, and tests software systems, applications, and solutions for various platforms.",
        "Skills": "Java, Python, C++, software design patterns, version control, problem-solving.",
        "PotentialCompanies": "Google, Microsoft, Amazon, Facebook, Apple, startups."
    },
    32: {
        "Role": "Technical Engineer",
        "Description": "Provides technical expertise for hardware and software systems, assisting with installation, configuration, and troubleshooting.",
        "Skills": "System installation, hardware troubleshooting, software configuration, technical support.",
        "PotentialCompanies": "Hewlett Packard, Dell, IBM, Wipro, HP."
    },
    33: {
        "Role": "Network Security Engineer",
        "Description": "Designs and implements secure networks to protect against cyber threats and unauthorized access.",
        "Skills": "Firewall management, network security protocols, VPN, intrusion detection systems, penetration testing.",
        "PotentialCompanies": "Cisco, Palo Alto Networks, Symantec, McAfee, IBM."
    },
    34: {
        "Role": "Software Quality Assurance (QA) / Testing",
        "Description": "Ensures software quality by testing and identifying defects before the product is released to users.",
        "Skills": "Manual testing, automation (Selenium), performance testing, test planning, bug tracking.",
        "PotentialCompanies": "Cognizant, Accenture, Wipro, Infosys, Capgemini."
    }
}
if __name__ == '__main__':
    app.run(debug=True)