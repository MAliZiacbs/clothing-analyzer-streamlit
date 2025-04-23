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


# ---- Color Mapping for Styling ----
# Simple mapping for common colors to CSS-compatible values
# Add more colors or hex codes as needed
COLOR_MAP = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "gold", # Use gold for yellow for better visibility
    "orange": "orange", "purple": "purple", "pink": "pink", "black": "black",
    "white": "black", # Display white text as black for visibility on white background
    "gray": "gray", "grey": "gray", "brown": "brown", "beige": "#F5F5DC", # Use hex for beige
    "teal": "teal", "cyan": "cyan", "navy": "navy", "maroon": "maroon",
    "olive": "olive", "lime": "lime", "silver": "silver", "gold": "gold",
    # Add light/dark variants if LLM commonly uses them
    "light blue": "lightblue", "dark blue": "darkblue", "light green": "lightgreen",
    "dark green": "darkgreen", "light gray": "lightgray", "dark gray": "darkgray",
    # Add more as needed...
}


# ---- Helper Function 1: Call BLIP Captioning Endpoint (Handles Dict/Str Response) ----
def call_blip_endpoint(image_bytes, blip_url, token):
    """
    Sends image bytes to the deployed BLIP captioning endpoint.
    Handles string or {'0': string} response format.
    Returns: (caption_string, error_string) - one will be None.
    """
    # Basic validation of config values provided
    if not blip_url or "YOUR_BLIP_ENDPOINT_URL_HERE" in blip_url: return None, "BLIP Endpoint URL missing."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token: return None, "Databricks API Token missing."

    print(f"Calling BLIP Endpoint: {blip_url}")
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_input = {"dataframe_split": {"columns": ["image_base64"], "data": [[b64_image]]}}
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    try:
        response = requests.post(blip_url, headers=headers, json=data_input, timeout=90)
        print(f"BLIP Endpoint Response Status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print(f"BLIP Raw Response Body: {result}")

        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            prediction_item = result['predictions'][0]
            caption = None
            if isinstance(prediction_item, str): caption = prediction_item
            elif isinstance(prediction_item, dict) and '0' in prediction_item and isinstance(prediction_item['0'], str):
                caption = prediction_item['0']; print("Note: Parsed caption from dict format")

            if caption and len(caption) > 3 and "error" not in caption.lower():
                print(f"BLIP Caption Received: '{caption}'"); return caption, None
            else: error_msg = f"Invalid caption/error from BLIP. Parsed: {caption}. Original: {prediction_item}"; print(error_msg); return None, error_msg
        else: error_msg = f"Unexpected BLIP response format: {result}"; print(error_msg); return None, error_msg

    except requests.exceptions.Timeout: error_msg = "Request to BLIP endpoint timed out."; print(error_msg); return None, error_msg
    except requests.exceptions.RequestException as e:
         error_detail = str(e);
         if e.response is not None: error_detail = f"Status {e.response.status_code}: {e.response.text[:200]}..."
         error_msg = f"API Request Failed (BLIP): {error_detail}"; print(error_msg); return None, error_msg
    except Exception as e: error_msg = f"Unexpected error calling BLIP: {str(e)}"; print(error_msg); traceback.print_exc(); return None, error_msg

# ---- Helper Function 2: Call LLM Endpoint (REVISED - Manual JSON List Parsing + Syntax Fix) ----
def call_llm_endpoint(caption_text, llm_endpoint_name, db_host, token):
    """
    Sends caption to Databricks LLM API. Asks for ALL items, expects JSON list within text response.
    Returns: (list_of_result_dicts, error_string) - one will be None.
    """
    # --- Config checks ---
    if not llm_endpoint_name or "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE" in llm_endpoint_name: return None, "LLM Endpoint Name missing."
    if not db_host or "YOUR_DATABRICKS_HOST_URL_HERE" in db_host: return None, "Databricks Host URL missing."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token: return None, "Databricks API Token missing."
    if not db_host.startswith(("http://", "https://")): return None, f"Databricks Host URL must start https://."

    # Configure OpenAI client
    try:
        print(f"Configuring OpenAI client for Databricks host: {db_host}")
        client = OpenAI(api_key=token, base_url=f"{db_host}/serving-endpoints")
    except Exception as e: return None, f"Failed to initialize LLM client: {str(e)}"

    # --- Prompt asking for JSON list ---
    prompt = f"""Analyze the clothing description generated from an image.
Identify ALL distinct clothing items mentioned and their corresponding colors.
Respond with ONLY a valid JSON list, where each object in the list represents one item and has 'color' and 'type' keys.
Use common clothing terms (e.g., 'shirt', 'pants', 'dress', 'jacket', 'suit', 'skirt', 't-shirt', 'boots').
If multiple items are mentioned, include all of them.
If color or type cannot be determined for an item, use the string "unknown".
Example response format: [{{"color": "blue", "type": "suit"}}, {{"color": "white", "type": "shirt"}}]

Description: "{caption_text}"

JSON List Output:"""

    print(f"Calling LLM Endpoint: {llm_endpoint_name}")
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": "You extract clothing items/colors as JSON list."}, {"role": "user", "content": prompt}],
            model=llm_endpoint_name, max_tokens=250, temperature=0.1
            # No response_format parameter here
        )
        # Check if response content is valid BEFORE parsing
        llm_response_content = None; finish_reason = "unknown"
        if chat_completion.choices and chat_completion.choices[0].message: llm_response_content = chat_completion.choices[0].message.content
        if chat_completion.choices and chat_completion.choices[0].finish_reason: finish_reason = chat_completion.choices[0].finish_reason
        print(f"LLM Raw Response String: {llm_response_content}"); print(f"LLM Finish Reason: {finish_reason}")

        # --- CORRECTED CHECK for None content ---
        if llm_response_content is None:
            error_msg = "LLM endpoint returned a null or empty response content."
            # Optionally add finish reason detail if available
            if finish_reason == 'content_filter':
                error_msg += " (Possibly due to content filtering)"
            elif finish_reason == 'length':
                error_msg += " (Possibly due to max_tokens limit)"
            print(error_msg) # Log the final error message
            return None, error_msg # Return the error clearly
        # --- END CORRECTION ---

        # --- Parse JSON list from text response ---
        json_list_str = None
        try:
            # Find the start and end of the JSON list within the response string
            start_index = llm_response_content.find('[')
            end_index = llm_response_content.rfind(']')

            if start_index != -1 and end_index != -1 and end_index >= start_index:
                json_list_str = llm_response_content[start_index : end_index + 1] # Extract list substring
                print(f"Extracted JSON list string: {json_list_str}")
                output_data = json.loads(json_list_str) # Parse the extracted string

                if isinstance(output_data, list):
                    # Validate items within the list
                    validated_list = []; valid_items_found = False
                    for item in output_data:
                        if isinstance(item, dict) and 'color' in item and 'type' in item:
                            item['color'] = str(item.get('color', 'unknown'))
                            item['type'] = str(item.get('type', 'unknown'))
                            validated_list.append(item); valid_items_found = True
                        else: print(f"Warning: Invalid item format: {item}")
                    if not valid_items_found and output_data: return None, f"LLM list items lacked keys: {output_data}"
                    print(f"LLM Parsed List: {validated_list}"); return validated_list, None # Success
                else: return None, f"Parsed data not list. Type: {type(output_data)}"
            else: return None, f"Could not find JSON list markers '[]' in: '{llm_response_content}'"
        except json.JSONDecodeError as json_e: error_msg = f"Failed to parse JSON list. Extracted: '{json_list_str if json_list_str else 'N/A'}'. Raw: '{llm_response_content}'. Error: {json_e}"; print(error_msg); return None, error_msg
        except Exception as parse_e: error_msg = f"Error processing LLM JSON: {parse_e}"; print(error_msg); traceback.print_exc(); return None, error_msg
        # --- END PARSING ---

    except Exception as e:
        # Catch API errors or other issues
        error_detail = str(e);
        if hasattr(e, 'status_code'): error_detail = f"Status {e.status_code}: {error_detail}"
        error_msg = f"LLM API Call Failed: {error_detail}"; print(error_msg)
        if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and 'message' in e.body: print(f"LLM API Error Message: {e.body['message']}")
        elif hasattr(e, 'message'): print(f"LLM API Error Message: {e.message}")
        traceback.print_exc(); return None, error_msg

# ---- Streamlit App User Interface ----
st.set_page_config(layout="wide", page_title="Clothing Analyzer")
st.title("👚👕👖 AI Clothing Analyzer")
st.markdown("Upload one or more images, I'll use Databricks AI (Vision + LLM) to identify color and type for each!")

# File uploader for Multiple Files
uploaded_files = st.file_uploader(
    "Choose clothing images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # Allow multiple files
)

# Process Multiple Files
if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} image(s)...")

    # Button to trigger analysis for ALL uploaded images
    if st.button("✨ Analyze All Uploaded Images"):
        # Iterate through each uploaded file object
        for file_index, uploaded_file in enumerate(uploaded_files):
            st.divider() # Add a separator between results for each image
            image_bytes = uploaded_file.getvalue()

            # Use columns for layout for each image
            col1, col2 = st.columns([1, 2]) # Image on left, results on right

            with col1:
                # Use container_width instead of deprecated use_column_width
                st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            # Process and display results in the second column
            # Use a unique key for the placeholder based on index to avoid issues
            display_area_placeholder = col2.empty()
            with display_area_placeholder.container():

                final_result_list = None
                error_message = None
                caption = None

                # --- Step 1: Call BLIP endpoint ---
                with st.spinner(f'Analyzing {uploaded_file.name}: Vision (1/2)...'):
                    caption, error_message = call_blip_endpoint(image_bytes, BLIP_ENDPOINT_URL, DATABRICKS_API_TOKEN)

                # --- Handle BLIP result ---
                if error_message:
                    st.error(f"**Vision Analysis Failed ({uploaded_file.name}):** {error_message}")
                elif caption:
                    st.info(f"Intermediate Caption:\n\"_{caption}_\"") # Show caption clearly

                    # --- Step 2: Call LLM endpoint ---
                    with st.spinner(f'Analyzing {uploaded_file.name}: LLM (2/2)...'):
                         final_result_list, error_message = call_llm_endpoint(caption, LLM_ENDPOINT_NAME, DATABRICKS_HOST, DATABRICKS_API_TOKEN)

                    # --- Handle LLM List Result (with new styling) ---
                    if error_message:
                         st.error(f"**LLM Analysis Failed ({uploaded_file.name}):** {error_message}")
                    elif final_result_list is not None:
                         st.subheader(f"Analysis Results ({uploaded_file.name}):")
                         if not final_result_list:
                              st.warning("LLM analysis didn't identify specific clothing items.")
                         else:
                              # Iterate and display with color styling
                              for item_index, item in enumerate(final_result_list):
                                  color_str = item.get('color', 'unknown')
                                  item_type_str = item.get('type', 'unknown').capitalize()
                                  color_str_lower = color_str.lower()

                                  # Get CSS color, fallback to default ('inherit')
                                  css_color = "inherit"
                                  if color_str_lower != "unknown":
                                       css_color = COLOR_MAP.get(color_str_lower, "inherit")

                                  # Construct display string
                                  if color_str_lower != "unknown":
                                       display_string = f"<span style='color:{css_color}; font-weight:bold;'>{color_str.capitalize()}</span> {item_type_str}"
                                  else:
                                       display_string = f"{item_type_str}" # Omit 'unknown' color

                                  st.markdown(f"- {display_string}", unsafe_allow_html=True)

                         # Optionally display raw JSON list
                         # with st.expander("Show Raw LLM JSON Response"):
                         #     st.json(final_result_list)
                    else:
                         st.error("LLM Analysis failed to return a valid result list or error.")
                else:
                     st.error("Vision analysis failed to return a caption or error.")


# --- Sidebar Info (Unchanged) ---
st.sidebar.header("Backend Info")
st.sidebar.markdown(f"""
This app uses a two-stage AI pipeline hosted on Databricks:
1.  A custom **Vision Model** (BLIP) endpoint provides an image caption.
2.  A **Foundation Model API** (LLM) endpoint extracts details from the caption.
""")
st.sidebar.caption(f"Vision Endpoint Used: ...{BLIP_ENDPOINT_URL[-40:]}")
st.sidebar.caption(f"LLM Endpoint Name Used: {LLM_ENDPOINT_NAME}")
st.sidebar.caption(f"Databricks Host Used: {DATABRICKS_HOST}")
st.sidebar.info("Configuration read from Streamlit Secrets or environment variables.")