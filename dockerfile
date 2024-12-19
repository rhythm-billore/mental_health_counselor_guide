FROM python:3.10-slim
COPY streamlit_app_counsellor_input /streamlit_app_counsellor_input
WORKDIR /streamlit_app_counsellor_input
RUN pip install -r requirements.txt
EXPOSE 8501
CMD streamlit run counsellor_input.py