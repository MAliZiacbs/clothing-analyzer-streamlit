import streamlit as st
from PIL import Image           # To display the image
import requests                 # To call the BLIP endpoint
import io                       # To handle image bytes
import base64                   # To encode image for BLIP endpoint
import os                       # To access environment variables (for local testing)
import json                     # To handle JSON data
from openai import OpenAI       # To call the Databricks Foundation LLM API
import traceback                # To print full errors during debugging

# ---- Configuration - GET THESE FROM DATABRICKS ----
# These values will be read from Streamlit Secrets when deployed (st.secrets.get)
# OR from environment variables (os.getenv) if set locally
# OR fallback to placeholder strings (edit directly ONLY for quick local test, remove before commit!)

# 1. BLIP Captioning Endpoint URL (Your custom endpoint from Phase 1, Step 1.10)
BLIP_ENDPOINT_URL = st.secrets.get("BLIP_ENDPOINT_URL", os.getenv("BLIP_ENDPOINT_URL", "YOUR_BLIP_ENDPOINT_URL_HERE"))

# 2. LLM Foundation Model API Endpoint Name (The built-in Databricks endpoint from Phase 2, Step 2.1)
LLM_ENDPOINT_NAME = st.secrets.get("LLM_ENDPOINT_NAME", os.getenv("LLM_ENDPOINT_NAME", "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE")) # e.g., "databricks-meta-llama-3-1-70b-instruct"

# 3. Databricks Host (Your workspace URL, e.g., "https://adb-....azuredatabricks.net")
#    Needed for the OpenAI client to target Databricks.
DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST", "YOUR_DATABRICKS_HOST_URL_HERE"))

# 4. Databricks Personal Access Token (PAT) (Needed for BOTH endpoints)
DATABRICKS_API_TOKEN = st.secrets.get("DATABRICKS_API_TOKEN", os.getenv("DATABRICKS_API_TOKEN", "YOUR_DATABRICKS_PAT_HERE"))


# ---- Helper Function 1: Call BLIP Captioning Endpoint ----
def call_blip_endpoint(image_bytes, blip_url, token):
    """
    Sends image bytes to the deployed BLIP captioning endpoint.
    Returns: (caption_string, error_string) - one will be None.
    """
    # Basic validation of config values provided
    if not blip_url or "YOUR_BLIP_ENDPOINT_URL_HERE" in blip_url:
        return None, "BLIP Endpoint URL is not configured in Streamlit app/secrets."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token:
        # Avoid leaking token existence in error messages for deployed apps
        if os.getenv("STREAMLIT_IS_DEPLOYED"): # Simple check if running in Streamlit Cloud
             return None, "Databricks API Token is not configured correctly."
        else:
             return None, "Databricks API Token is not configured (check secrets/env vars/placeholders)."

    print(f"Calling BLIP Endpoint: {blip_url}") # Print for server-side debugging (won't show in browser)
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    # Input format matching the signature defined for the BLIP model
    data_input = {"dataframe_split": {"columns": ["image_base64"], "data": [[b64_image]]}}
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    try:
        response = requests.post(blip_url, headers=headers, json=data_input, timeout=90) # 90 sec timeout
        print(f"BLIP Endpoint Response Status: {response.status_code}") # Debugging
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        result = response.json()
        print(f"BLIP Raw Response Body: {result}") # Debugging

        # Extract prediction (should be a caption string)
        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            caption = result['predictions'][0]
            # Basic check if caption seems valid
            if isinstance(caption, str) and len(caption) > 3 and "error" not in caption.lower():
                print(f"BLIP Caption Received: '{caption}'")
                return caption, None # Success: return caption, no error
            else:
                 error_msg = f"Received invalid caption or error message from BLIP endpoint: {caption}"
                 print(error_msg)
                 return None, error_msg # Failure: return None, error message
        else:
            error_msg = f"Unexpected response format from BLIP endpoint (missing 'predictions'): {result}"
            print(error_msg)
            return None, error_msg # Failure: return None, error message

    except requests.exceptions.Timeout:
         error_msg = "Request to BLIP endpoint timed out."
         print(error_msg)
         return None, error_msg
    except requests.exceptions.RequestException as e:
         # Try to get more specific error info
         error_detail = str(e)
         if e.response is not None:
              error_detail = f"Status {e.response.status_code}: {e.response.text[:200]}..."
         error_msg = f"API Request Failed (BLIP): {error_detail}"
         print(error_msg)
         return None, error_msg
    except Exception as e:
         # Catch-all for other unexpected errors
         error_msg = f"Unexpected error calling BLIP endpoint: {str(e)}"
         print(error_msg)
         traceback.print_exc() # Print full traceback to server logs
         return None, error_msg

# ---- Helper Function 2: Call LLM Foundation Model API Endpoint ----
def call_llm_endpoint(caption_text, llm_endpoint_name, db_host, token):
    """
    Sends the caption text to the Databricks Foundation Model API for analysis.
    Returns: (result_dict, error_string) - one will be None.
    """
    # Basic validation of config values
    if not llm_endpoint_name or "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE" in llm_endpoint_name:
        return None, "LLM Endpoint Name is not configured in Streamlit app/secrets."
    if not db_host or "YOUR_DATABRICKS_HOST_URL_HERE" in db_host:
        return None, "Databricks Host URL is not configured in Streamlit app/secrets."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token:
        if os.getenv("STREAMLIT_IS_DEPLOYED"):
             return None, "Databricks API Token is not configured correctly."
        else:
             return None, "Databricks API Token is not configured (check secrets/env vars/placeholders)."

    # Configure OpenAI client to point to Databricks Foundation Model API
    try:
        print(f"Configuring OpenAI client for Databricks host: {db_host}")
        client = OpenAI(
            api_key=token,                      # Your Databricks PAT
            base_url=f"{db_host}/serving-endpoints" # Databricks API base URL
            )
    except Exception as e:
        error_msg = f"Failed to initialize LLM client: {str(e)}"
        print(error_msg)
        return None, error_msg

    # Construct the prompt for the LLM, asking for JSON output
    prompt = f"""Analyze the clothing description generated from an image.
Identify the primary color and the type of the main clothing item described.
Consider common clothing terms (shirt, pants, dress, jacket, sweater, skirt, top, t-shirt, jeans, trousers, etc.).
Respond ONLY with a valid JSON object containing 'color' and 'type' keys.
If color or type cannot be determined, use "unknown". Example: {{"color": "blue", "type": "jacket"}}

Description: "{caption_text}"

JSON Output:"""

    print(f"Calling LLM Endpoint: {llm_endpoint_name}")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant expert at extracting clothing color and type from descriptions. You output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=llm_endpoint_name,           # The Databricks endpoint name (e.g., databricks-meta-llama-3-1-70b-instruct)
            max_tokens=100,                    # Limit response size
            temperature=0.1,                   # Low temp for factual extraction
            response_format={"type": "json_object"} # Request JSON output format
        )
        llm_response_content = chat_completion.choices[0].message.content
        print(f"LLM Raw Response String: {llm_response_content}") # Debugging

        # Attempt to parse the JSON response string
        try:
            output_json = json.loads(llm_response_content)
            # Basic validation of the parsed JSON
            if isinstance(output_json, dict) and 'color' in output_json and 'type' in output_json:
                # Ensure values are strings for consistency
                output_json['color'] = str(output_json.get('color', 'unknown'))
                output_json['type'] = str(output_json.get('type', 'unknown'))
                print(f"LLM Parsed JSON: {output_json}")
                return output_json, None # Success: return parsed dict, no error
            else:
                 error_msg = f"LLM JSON response missing required keys ('color', 'type'): {output_json}"
                 print(error_msg)
                 return None, error_msg # Failure: return None, error message
        except json.JSONDecodeError as json_e:
            error_msg = f"Failed to parse LLM response as JSON. Raw: '{llm_response_content}'. Error: {json_e}"
            print(error_msg)
            return None, error_msg # Failure: return None, error message

    except Exception as e:
        # Catch API errors (like 4xx, 5xx from Databricks endpoint) or other issues
        error_detail = str(e)
        # Try to extract specific error code/message if available
        if hasattr(e, 'status_code'): error_detail = f"Status {e.status_code}: {error_detail}"
        error_msg = f"LLM API Call Failed: {error_detail}"
        print(error_msg)
        # Optionally print more details from the error object if available
        if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and 'message' in e.body:
             print(f"LLM API Error Message: {e.body['message']}")
        elif hasattr(e, 'message'):
             print(f"LLM API Error Message: {e.message}")

        traceback.print_exc() # Print traceback for unexpected errors
        return None, error_msg # Failure: return None, error message

# ---- Streamlit App User Interface ----
st.set_page_config(layout="centered", page_title="Clothing Analyzer")
st.title("👚👕👖 AI Clothing Analyzer")
st.markdown("Upload an image, I'll use Databricks AI (Vision + LLM) to tell you the item's color and type!")

# File uploader
uploaded_file = st.file_uploader("Choose a clothing image...", type=["jpg", "jpeg", "png"])

# Area for displaying results and errors
results_area = st.container()

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    results_area.image(image_bytes, caption='Uploaded Image', width=300)

    if results_area.button("✨ Analyze Clothing"):
        # Clear previous results/errors from the display area
        results_area.empty()
        display_area = results_area.container() # Use a sub-container to write new results

        final_result = None
        error_message = None
        caption = None

        # --- Step 1: Call BLIP endpoint ---
        with st.spinner('Analyzing... Calling Vision Service (Step 1/2)'):
            caption, error_message = call_blip_endpoint(image_bytes, BLIP_ENDPOINT_URL, DATABRICKS_API_TOKEN)

        # --- Handle BLIP result ---
        if error_message:
            display_area.error(f"**Vision Analysis Failed:** {error_message}")
        elif caption:
            display_area.info(f"Intermediate Caption: \"{caption}\"") # Show caption

            # --- Step 2: Call LLM endpoint ---
            with st.spinner('Analyzing... Calling LLM Service (Step 2/2)'):
                 final_result, error_message = call_llm_endpoint(caption, LLM_ENDPOINT_NAME, DATABRICKS_HOST, DATABRICKS_API_TOKEN)

            # --- Handle LLM result ---
            if error_message:
                 display_area.error(f"**LLM Analysis Failed:** {error_message}")
            elif final_result:
                 display_area.subheader("Analysis Result:")
                 color = final_result.get('color', 'N/A') # Use .get for safer access
                 item_type = final_result.get('type', 'N/A')
                 display_area.success(f"**Detected Color:** {color.capitalize()}")
                 display_area.success(f"**Detected Type:** {item_type.capitalize()}")
                 # Optionally display raw JSON
                 # with display_area.expander("Show Raw LLM JSON"):
                 #     st.json(final_result)
            else:
                 # This case should ideally be caught by error handling in call_llm_endpoint
                 display_area.error("LLM Analysis seemed to complete but no valid result dictionary was obtained.")
        else:
             # This case should ideally be caught by error handling in call_blip_endpoint
             display_area.error("Vision analysis did not return a caption or an error.")


# --- Sidebar Info ---
st.sidebar.header("Backend Info")
st.sidebar.markdown(f"""
This app uses a two-stage AI pipeline hosted on Databricks:
1.  A custom **Vision Model** (BLIP) endpoint provides an image caption.
2.  A **Foundation Model API** (LLM) endpoint extracts details from the caption.
""")
# Show partial URLs/names for info without leaking full secrets
st.sidebar.caption(f"Vision Endpoint Used: ...{BLIP_ENDPOINT_URL[-40:]}")
st.sidebar.caption(f"LLM Endpoint Name Used: {LLM_ENDPOINT_NAME}")
st.sidebar.caption(f"Databricks Host Used: {DATABRICKS_HOST}")
st.sidebar.info("Configuration read from Streamlit Secrets (if deployed) or environment variables/placeholders (if local).")