import streamlit as st
from PIL import Image
import requests
import io
import base64
import os
import json # Import json library

# --- Configuration ---
# !!! IMPORTANT: Replace this placeholder with your ACTUAL Databricks Serving Endpoint URL (from Phase 2) !!!
DATABRICKS_ENDPOINT_URL = "https://adb-360063509637705.5.azuredatabricks.net/serving-endpoints/clothing-analyzer-prod/invocations"

# --- Security Best Practice ---
# Use Streamlit Secrets for your API Token when deploying!
# For local testing ONLY, you can set an environment variable (recommended)
# or paste your token here temporarily (less secure).
# How to set secrets in Streamlit Cloud (Phase 4):
# 1. Go to your app's settings on share.streamlit.io -> Secrets
# 2. Paste: DATABRICKS_API_TOKEN="dapiXXXXXXXXXXXXXXXXXXXX" (replace with your actual token)
DATABRICKS_API_TOKEN = st.secrets.get("DATABRICKS_API_TOKEN", os.getenv("DATABRICKS_API_TOKEN", "YOUR_DATABRICKS_TOKEN_FOR_LOCAL_TEST_ONLY"))


# --- Helper Function to Call Databricks ---
def call_databricks_endpoint(image_bytes):
    """Sends image to the configured Databricks Model Serving Endpoint"""

    # Basic validation of config
    if not DATABRICKS_ENDPOINT_URL or "YOUR_DATABRICKS_SERVING_ENDPOINT_URL_HERE" in DATABRICKS_ENDPOINT_URL:
        st.error("Databricks Endpoint URL is not configured correctly in the app.py script.")
        return {"error": "Endpoint URL not set."} # Return error dict
    if not DATABRICKS_API_TOKEN or "YOUR_DATABRICKS_TOKEN_FOR_LOCAL_TEST_ONLY" in DATABRICKS_API_TOKEN:
        st.error("Databricks API Token is not configured. Check Streamlit Secrets (for deployed app) or environment variables (for local).")
        return {"error": "API Token not set."} # Return error dict

    # Encode image bytes as Base64 string
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Prepare the input data in the standard format expected by MLflow PyFunc / Databricks Model Serving
    # This matches the signature defined: DataFrame with 'image_base64' column
    data_input = {
        "dataframe_split": {
            "columns": ["image_base64"],
            "data": [[b64_image]]
        }
    }
    data_json = json.dumps(data_input)

    headers = {
        'Authorization': f'Bearer {DATABRICKS_API_TOKEN}',
        'Content-Type': 'application/json'
    }

    st.write(f"Sending request to: {DATABRICKS_ENDPOINT_URL}") # Log for debugging
    try:
        response = requests.post(DATABRICKS_ENDPOINT_URL, headers=headers, data=data_json, timeout=120) # Generous timeout
        st.write(f"Response Status Code: {response.status_code}") # Log for debugging
        response.raise_for_status() # Checks for HTTP errors (like 401, 403, 404, 500)

        result = response.json()
        st.write(f"Raw Response Body: {result}") # Log for debugging

        # Parse the response - MLflow pyfunc predictions are typically in a 'predictions' list
        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            # The prediction should be the JSON string returned by our PyFunc
            prediction_json_str = result['predictions'][0]
            try:
                # Parse the JSON string into a Python dictionary
                output_data = json.loads(prediction_json_str)
                # Check if the parsed data itself contains an error reported by the pyfunc
                if isinstance(output_data, dict) and output_data.get("error"):
                     st.warning(f"Backend function reported an error: {output_data['error']}")
                     # Optionally include description if available
                     if output_data.get("description"):
                          st.info(f"Vision Description (if available): {output_data['description']}")
                     if output_data.get("raw_output"):
                           st.text_area("LLM Raw Output (if available):", output_data['raw_output'], height=100)
                     return {"error": f"Analysis failed internally: {output_data['error']}"}

                return output_data # Return the dictionary on success

            except json.JSONDecodeError:
                 st.warning(f"Could not parse the prediction string from endpoint as JSON: {prediction_json_str}")
                 return {"error": "Failed to parse result JSON from endpoint", "raw_prediction": prediction_json_str}
        else:
            st.warning(f"Unexpected response format from endpoint (missing 'predictions' list): {result}")
            return {"error": "Received unexpected response format from endpoint."}

    except requests.exceptions.Timeout:
         st.error("The request to the Databricks endpoint timed out. The backend might be overloaded or the model is taking too long.")
         return {"error": "Request timed out."}
    except requests.exceptions.HTTPError as http_err:
         st.error(f"API Request Failed with HTTP error: {http_err}")
         st.error(f"Response Body: {http_err.response.text[:500]}...") # Show beginning of error message
         return {"error": f"API returned status {http_err.response.status_code}"}
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed with connection/other error: {e}")
        return {"error": "API connection failed."}
    except Exception as e:
        st.error(f"An unexpected error occurred when calling the endpoint: {e}")
        import traceback
        st.text(traceback.format_exc()) # Show full traceback for debugging unexpected errors
        return {"error": f"Frontend processing error: {str(e)}"}


# --- Streamlit App User Interface ---
st.set_page_config(layout="wide", page_title="Clothing Analyzer")
st.title("👚👕👖👗 AI Clothing Analyzer")
st.markdown("Upload an image of a clothing item. A backend AI pipeline on Databricks (using Vision + LLM) will identify its color and type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.getvalue()
    # Display the uploaded image
    st.image(image_bytes, caption='Uploaded Image', use_column_width=False, width=300)

    st.divider()

    # Button to trigger analysis
    if st.button("Analyze Clothing"):
        # Show a spinner while processing
        with st.spinner('🧠 Analyzing image via Databricks... This may take ~15-60 seconds.'):
            analysis_result = call_databricks_endpoint(image_bytes) # Returns a dictionary

        st.subheader("Analysis Result:")
        # Check if the result is a dictionary and doesn't contain an 'error' key
        if isinstance(analysis_result, dict) and 'error' not in analysis_result:
            color = analysis_result.get('color', 'N/A') # Use .get for safer access
            item_type = analysis_result.get('type', 'N/A')
            st.success(f"**Detected Color:** {color.capitalize()}")
            st.success(f"**Detected Type:** {item_type.capitalize()}")
            # Optionally display the raw JSON received
            with st.expander("Show Raw JSON Response from Backend"):
                st.json(analysis_result)
        elif isinstance(analysis_result, dict) and 'error' in analysis_result:
             # Display specific error already logged inside call_databricks_endpoint
             st.error(f"**Analysis failed:** {analysis_result['error']}")
             if 'raw_prediction' in analysis_result:
                  st.text_area("Raw Prediction String (if available):", analysis_result['raw_prediction'], height=100)
        else:
             # Fallback for unexpected results (e.g., not a dict)
             st.error("An unexpected result format was received from the backend.")
             st.write("Raw response data:", analysis_result)

# Sidebar for configuration information
st.sidebar.header("Configuration Info")
st.sidebar.info("This app calls a custom LLM Agent deployed on Databricks Model Serving.")
st.sidebar.caption(f"Endpoint URL configured: ...{DATABRICKS_ENDPOINT_URL[-50:]}") # Show partial URL
st.sidebar.markdown("---")
st.sidebar.caption("Developed using Databricks & Streamlit.")