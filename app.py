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

# 3. Databricks Host (Your workspace URL, INCLUDING https://)
DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST", "https://YOUR_DATABRICKS_HOST_URL_HERE"))

# 4. Databricks Personal Access Token (PAT) (Needed for BOTH endpoints)
DATABRICKS_API_TOKEN = st.secrets.get("DATABRICKS_API_TOKEN", os.getenv("DATABRICKS_API_TOKEN", "dapi_YOUR_DATABRICKS_PAT_HERE"))


# ---- Helper Function 1: Call BLIP Captioning Endpoint (Handles Dict Response) ----
def call_blip_endpoint(image_bytes, blip_url, token):
    """
    Sends image bytes to the deployed BLIP captioning endpoint.
    Handles string or {'0': string} response format.
    Returns: (caption_string, error_string) - one will be None.
    """
    # Basic validation of config values provided
    if not blip_url or "YOUR_BLIP_ENDPOINT_URL_HERE" in blip_url:
        return None, "BLIP Endpoint URL is not configured in Streamlit app/secrets."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token:
        if os.getenv("STREAMLIT_IS_DEPLOYED"): return None, "Databricks API Token not configured correctly."
        else: return None, "Databricks API Token not configured (check secrets/env vars)."

    print(f"Calling BLIP Endpoint: {blip_url}") # Server-side log
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_input = {"dataframe_split": {"columns": ["image_base64"], "data": [[b64_image]]}}
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    try:
        response = requests.post(blip_url, headers=headers, json=data_input, timeout=90)
        print(f"BLIP Endpoint Response Status: {response.status_code}") # Server-side log
        response.raise_for_status()
        result = response.json()
        print(f"BLIP Raw Response Body: {result}") # Server-side log

        # Modified BLIP Response Parsing
        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            prediction_item = result['predictions'][0]
            caption = None
            if isinstance(prediction_item, str): caption = prediction_item
            elif isinstance(prediction_item, dict) and '0' in prediction_item and isinstance(prediction_item['0'], str):
                caption = prediction_item['0']
                print("Note: Parsed caption from dictionary format {'0': caption}") # Server-side log

            if caption and len(caption) > 3 and "error" not in caption.lower():
                print(f"BLIP Caption Received: '{caption}'") # Server-side log
                return caption, None # Success
            else:
                 error_msg = f"Received invalid caption or error message from BLIP endpoint. Parsed value: {caption}. Original item: {prediction_item}"
                 print(error_msg)
                 return None, error_msg
        else:
            error_msg = f"Unexpected response format from BLIP endpoint (missing 'predictions' or empty list): {result}"
            print(error_msg)
            return None, error_msg

    except requests.exceptions.Timeout:
         error_msg = "Request to BLIP endpoint timed out."
         print(error_msg)
         return None, error_msg
    except requests.exceptions.RequestException as e:
         error_detail = str(e)
         if e.response is not None: error_detail = f"Status {e.response.status_code}: {e.response.text[:200]}..."
         error_msg = f"API Request Failed (BLIP): {error_detail}"
         print(error_msg)
         return None, error_msg
    except Exception as e:
         error_msg = f"Unexpected error calling BLIP endpoint: {str(e)}"
         print(error_msg)
         traceback.print_exc()
         return None, error_msg

# ---- Helper Function 2: Call LLM Foundation Model API Endpoint (Includes None Check) ----
def call_llm_endpoint(caption_text, llm_endpoint_name, db_host, token):
    """
    Sends the caption text to the Databricks Foundation Model API for analysis.
    Asks for ALL items and expects a JSON list as output.
    Returns: (list_of_result_dicts, error_string) - one will be None.
    """
    # --- Config checks ---
    if not llm_endpoint_name or "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE" in llm_endpoint_name:
        return None, "LLM Endpoint Name is not configured in Streamlit app/secrets."
    if not db_host or "YOUR_DATABRICKS_HOST_URL_HERE" in db_host:
        return None, "Databricks Host URL is not configured or missing 'https://'."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token:
        if os.getenv("STREAMLIT_IS_DEPLOYED"): return None, "Databricks API Token not configured."
        else: return None, "Databricks API Token not configured (check secrets/env vars)."
    if not db_host.startswith(("http://", "https://")):
         return None, f"Databricks Host URL ('{db_host}') must start with 'https://' or 'http://'."

    # Configure OpenAI client
    try:
        print(f"Configuring OpenAI client for Databricks host: {db_host}")
        client = OpenAI(api_key=token, base_url=f"{db_host}/serving-endpoints")
    except Exception as e:
        error_msg = f"Failed to initialize LLM client: {str(e)}"
        print(error_msg)
        return None, error_msg

    # Construct the prompt
    prompt = f"""Analyze the clothing description generated from an image.
Identify ALL distinct clothing items mentioned and their corresponding colors.
Respond ONLY with a valid JSON list, where each object in the list represents one item and has 'color' and 'type' keys.
Use common clothing terms (e.g., 'shirt', 'pants', 'dress', 'jacket', 'suit', 'skirt', 't-shirt', 'boots').
If multiple items are mentioned, include all of them.
If color or type cannot be determined for an item, use the string "unknown".
Example response format: [{{"color": "blue", "type": "suit"}}, {{"color": "white", "type": "shirt"}}]

Description: "{caption_text}"

JSON List Output:"""

    print(f"Calling LLM Endpoint: {llm_endpoint_name}")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant expert at extracting ALL clothing items and their colors from descriptions. You output only a valid JSON list of objects, where each object has 'color' and 'type' keys."},
                {"role": "user", "content": prompt}
            ],
            model=llm_endpoint_name,
            max_tokens=250,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        # --- Check if response content is valid BEFORE parsing ---
        llm_response_content = None
        if chat_completion.choices and chat_completion.choices[0].message:
             llm_response_content = chat_completion.choices[0].message.content

        print(f"LLM Raw Response String: {llm_response_content}") # Debugging

        # --- ADDED CHECK for None content ---
        if llm_response_content is None:
            error_msg = "LLM endpoint returned a null or empty response content."
            print(error_msg)
            # Consider checking finish reason if available: chat_completion.choices[0].finish_reason
            finish_reason = chat_completion.choices[0].finish_reason if chat_completion.choices else "unknown"
            print(f"LLM Finish Reason: {finish_reason}") # e.g., 'stop', 'length', 'content_filter'
            if finish_reason == 'content_filter':
                error_msg += " (Possibly due to content filtering)"
            return None, error_msg
        # --- END ADDED CHECK ---

        # Attempt to parse the JSON response string
        try:
            output_data = json.loads(llm_response_content)

            # Check if the output is a list
            if isinstance(output_data, list):
                validated_list = []
                valid_items_found = False
                for item in output_data:
                    if isinstance(item, dict) and 'color' in item and 'type' in item:
                        item['color'] = str(item.get('color', 'unknown'))
                        item['type'] = str(item.get('type', 'unknown'))
                        validated_list.append(item)
                        valid_items_found = True
                    else: print(f"Warning: Invalid item format in LLM list: {item}")

                if not valid_items_found and output_data:
                     error_msg = f"LLM list items lacked 'color'/'type' keys: {output_data}"
                     print(error_msg); return None, error_msg

                print(f"LLM Parsed List: {validated_list}")
                return validated_list, None # Success

            elif isinstance(output_data, dict) and 'error' in output_data:
                 error_msg = f"LLM returned error object: {output_data['error']}"; print(error_msg); return None, error_msg
            else:
                 error_msg = f"LLM response not a list. Type: {type(output_data)}, Value: {output_data}"; print(error_msg); return None, error_msg

        except json.JSONDecodeError as json_e:
            error_msg = f"Failed to parse LLM response as JSON. Raw: '{llm_response_content}'. Error: {json_e}"; print(error_msg); return None, error_msg
        except Exception as parse_e:
             error_msg = f"Error processing LLM JSON response: {parse_e}"; print(error_msg); traceback.print_exc(); return None, error_msg

    except Exception as e:
        error_detail = str(e)
        if hasattr(e, 'status_code'): error_detail = f"Status {e.status_code}: {error_detail}"
        error_msg = f"LLM API Call Failed: {error_detail}"
        print(error_msg)
        if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and 'message' in e.body: print(f"LLM API Error Message: {e.body['message']}")
        elif hasattr(e, 'message'): print(f"LLM API Error Message: {e.message}")
        traceback.print_exc()
        return None, error_msg

# ---- Streamlit App User Interface ----
st.set_page_config(layout="centered", page_title="Clothing Analyzer")
st.title("👚👕👖 AI Clothing Analyzer")
st.markdown("Upload an image, I'll use Databricks AI (Vision + LLM) to tell you the item's color and type!")

# File uploader
uploaded_file = st.file_uploader("Choose a clothing image...", type=["jpg", "jpeg", "png"])

# Area for displaying image and results
results_area = st.container()

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    # Display uploaded image within the results area
    results_area.image(image_bytes, caption='Uploaded Image', width=300)

    if results_area.button("✨ Analyze Clothing"):
        # Clear previous results/errors from the display area below the button
        # Using an empty container allows replacing content without scroll jump
        display_area_placeholder = results_area.empty()

        final_result_list = None
        error_message = None
        caption = None

        # --- Step 1: Call BLIP endpoint ---
        with st.spinner('Analyzing... Calling Vision Service (Step 1/2)'):
            caption, error_message = call_blip_endpoint(image_bytes, BLIP_ENDPOINT_URL, DATABRICKS_API_TOKEN)

        # --- Handle BLIP result ---
        with display_area_placeholder.container(): # Write results into the placeholder
            if error_message:
                st.error(f"**Vision Analysis Failed:** {error_message}")
            elif caption:
                st.info(f"Intermediate Caption: \"{caption}\"") # Show caption

                # --- Step 2: Call LLM endpoint ---
                with st.spinner('Analyzing... Calling LLM Service (Step 2/2)'):
                     # Expecting a list of dicts now, or None if error
                     final_result_list, error_message = call_llm_endpoint(caption, LLM_ENDPOINT_NAME, DATABRICKS_HOST, DATABRICKS_API_TOKEN)

                # --- Handle LLM List Result ---
                if error_message:
                     st.error(f"**LLM Analysis Failed:** {error_message}")
                elif final_result_list is not None: # Check if we got a list (could be empty)
                     st.subheader("Analysis Results:")
                     if not final_result_list: # Handle empty list
                          st.warning("LLM analysis didn't identify specific clothing items from the caption.")
                     else:
                          # Iterate through the list of dictionaries and display each item
                          for item_index, item in enumerate(final_result_list):
                              color = item.get('color', 'N/A').capitalize()
                              item_type = item.get('type', 'N/A').capitalize()
                              # Use markdown for slightly nicer formatting
                              st.markdown(f"- **Item {item_index+1}:** {color} {item_type}")

                     # Optionally display raw JSON list
                     # with st.expander("Show Raw LLM JSON Response"):
                     #     st.json(final_result_list)
                else:
                     st.error("LLM Analysis seemed to complete but no valid result list or error was returned.")

            else:
                 st.error("Vision analysis did not return a caption or an error.")


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