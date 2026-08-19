"""
Loan Application Analyzer
Parallel Chains + Merge Chain

GenAI-powered loan analysis using LangChain, OpenAI LLMs, and Streamlit.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Initialize OpenAI LLM
model = ChatOpenAI()

parser = StrOutputParser()

# 1. Credit Risk Assessment
credit_risk_prompt = PromptTemplate(
    template="""
    You are a banking credit risk analyst.

    Analyze the following loan application and assess the credit risk.
    Classify risk as Low, Medium, or High with a short justification.

    Loan Application:
    {application}
    """,
    input_variables=["application"],
    validate_template=True,
)

# 2. Fraud Detection
fraud_prompt = PromptTemplate(
    template="""
    You are a bank fraud detection expert.

    Analyze the loan application and identify any potential fraud signals
    such as inconsistencies, unusual claims, or red flags.

    Loan Application:
    {application}
    """,
    input_variables=["application"],
    validate_template=True,
)

# 3. Eligibility Assessment
eligibility_prompt = PromptTemplate(
    template="""
    You are a loan approval officer.

    Based on the loan application, decide whether the applicant is:
    - Eligible
    - Conditionally Eligible
    - Not Eligible

    Provide a short reason.

    Loan Application:
    {application}
    """,
    input_variables=["application"],
    validate_template=True,
)

# 4. Interest Rate Recommendation
interest_prompt = PromptTemplate(
    template="""
    You are a bank pricing strategist.

    Based on the applicant’s profile, recommend an interest rate range
    and explain the reasoning briefly.

    Loan Application:
    {application}
    """,
    input_variables=["application"],
    validate_template=True,
)

# 5. Customer Explanation
customer_prompt = PromptTemplate(
    template="""
    You are a customer relationship manager.

    Explain the loan decision in simple, non-technical language
    so a customer can easily understand it.

    Loan Application:
    {application}
    """,
    input_variables=["application"],
    validate_template=True,
)

# Final Merge / Synthesis Prompt
merge_prompt = PromptTemplate(
    template="""
    You are a senior loan approval committee member.

    Based on the following expert assessments, make a FINAL loan decision.

    Credit Risk Assessment:
    {credit_risk}

    Fraud Analysis:
    {fraud_check}

    Eligibility Decision:
    {eligibility}

    Interest Rate Recommendation:
    {interest_rate}

    Customer Explanation Summary:
    {customer_explanation}

    Your task:
    1. Give final decision: Approved / Conditionally Approved / Rejected
    2. Provide a concise justification
    3. Mention any conditions if applicable
    4. Give a final recommended interest rate range
    """,
    input_variables=[
        "credit_risk",
        "fraud_check",
        "eligibility",
        "interest_rate",
        "customer_explanation",
    ],
    validate_template=True,
)

# Execute five expert assessments in parallel
parallel_chain = RunnableParallel(
    {
        "credit_risk": credit_risk_prompt | model | parser,
        "fraud_check": fraud_prompt | model | parser,
        "eligibility": eligibility_prompt | model | parser,
        "interest_rate": interest_prompt | model | parser,
        "customer_explanation": customer_prompt | model | parser,
    }
)

# Sequential merge/synthesis chain
merge_chain = merge_prompt | model | parser

# Complete pipeline: parallel assessments -> final synthesis
chain = parallel_chain | merge_chain


# Streamlit UI
st.title("Loan Application Analyzer")

loan_amount = st.text_input("Enter the loan amount.")
monthly_income = st.text_input("Enter your monthly salary.")
profession = st.text_input("Enter your profession.")
tenure = st.selectbox("Select your tenure", [0, 1, 2, 3, 4, 5, 6, 7])
credit_score = st.text_input("Enter your credit score.")
loan_defaulter = st.selectbox("Loan defaulter", ["Yes", "No"])

loan_application_text = f"""
Applicant is requesting a personal loan of INR {loan_amount}.
Monthly income is INR {monthly_income}.
Profession: {profession}.
Loan tenure: {tenure} years.
Credit score: {credit_score}.
Previous loan defaulter: {loan_defaulter}.
"""

if st.button("Submit Application"):
    # Basic validation
    if not loan_amount or not monthly_income or not profession or not credit_score:
        st.warning("Please complete all required fields.")
    else:
        with st.spinner("Analyzing loan application..."):
            try:
                result = chain.invoke(
                    {"application": loan_application_text}
                )

                st.success("Analysis Complete")
                st.write(result)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
