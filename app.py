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
BLIP_ENDPOINT_URL = st.secrets.get("BLIP_ENDPOINT_URL", os.getenv("BLIP_ENDPOINT_URL", "YOUR_BLIP_ENDPOINT_URL_HERE"))
LLM_ENDPOINT_NAME = st.secrets.get("LLM_ENDPOINT_NAME", os.getenv("LLM_ENDPOINT_NAME", "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE"))
DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST", "https://YOUR_DATABRICKS_HOST_URL_HERE"))
DATABRICKS_API_TOKEN = st.secrets.get("DATABRICKS_API_TOKEN", os.getenv("DATABRICKS_API_TOKEN", "dapi_YOUR_DATABRICKS_PAT_HERE"))

# ---- Color Mapping for Styling (Unchanged) ----
COLOR_MAP = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "gold",
    "orange": "orange", "purple": "purple", "pink": "pink", "black": "black",
    "white": "black", "gray": "gray", "grey": "gray", "brown": "brown",
    "beige": "#F5F5DC", "teal": "teal", "cyan": "cyan", "navy": "navy",
    "maroon": "maroon", "olive": "olive", "lime": "lime", "silver": "silver",
    "gold": "gold", "light blue": "lightblue", "dark blue": "darkblue",
    "light green": "lightgreen", "dark green": "darkgreen", "light gray": "lightgray",
    "dark gray": "darkgray",
}

# ---- Helper Function 1: Call BLIP Captioning Endpoint (Unchanged) ----
def call_blip_endpoint(image_bytes, blip_url, token):
    # (Code for call_blip_endpoint remains the same as the previous version)
    if not blip_url or "YOUR_BLIP_ENDPOINT_URL_HERE" in blip_url: return None, "BLIP URL missing."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token: return None, "Token missing."
    print(f"Calling BLIP Endpoint: {blip_url}")
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_input = {"dataframe_split": {"columns": ["image_base64"], "data": [[b64_image]]}}
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    try:
        response = requests.post(blip_url, headers=headers, json=data_input, timeout=90)
        response.raise_for_status(); result = response.json(); print(f"BLIP Raw: {result}")
        if 'predictions' in result and isinstance(result['predictions'], list) and len(result['predictions']) > 0:
            item = result['predictions'][0]; caption = None
            if isinstance(item, str): caption = item
            elif isinstance(item, dict) and '0' in item and isinstance(item['0'], str): caption = item['0']
            if caption and len(caption) > 3 and "error" not in caption.lower(): return caption, None
            else: return None, f"Invalid caption/error from BLIP. Parsed: {caption}. Original: {item}"
        else: return None, f"Unexpected BLIP response format: {result}"
    except requests.exceptions.Timeout: return None, "Request to BLIP timed out."
    except requests.exceptions.RequestException as e: error_detail = str(e); print(f"BLIP API Error: {error_detail}"); return None, f"API Error (BLIP): {error_detail}"
    except Exception as e: print(f"BLIP Call Error: {e}"); traceback.print_exc(); return None, f"BLIP Call Error: {e}"


# ---- Helper Function 2: Call LLM Endpoint (Unchanged) ----
def call_llm_endpoint(caption_text, llm_endpoint_name, db_host, token):
    """ Sends caption to Databricks LLM API. Expects JSON list in text response. """
    # (Code for call_llm_endpoint remains the same as the previous version)
    if not llm_endpoint_name or "YOUR_LLM_FOUNDATION_MODEL_NAME_HERE" in llm_endpoint_name: return None, "LLM Name missing."
    if not db_host or "YOUR_DATABRICKS_HOST_URL_HERE" in db_host: return None, "DB Host missing."
    if not token or "YOUR_DATABRICKS_PAT_HERE" in token: return None, "Token missing."
    if not db_host.startswith(("http://", "https://")): return None, f"DB Host URL must start https://."
    try: client = OpenAI(api_key=token, base_url=f"{db_host}/serving-endpoints")
    except Exception as e: return None, f"Failed to init LLM client: {e}"
    prompt = f"""Analyze the clothing description generated from an image. Identify ALL distinct clothing items mentioned and their corresponding colors. Respond with ONLY a valid JSON list, where each object in the list represents one item and has 'color' and 'type' keys. Use common clothing terms. If color or type cannot be determined use "unknown". Example: [{{"color": "blue", "type": "suit"}}, {{"color": "white", "type": "shirt"}}] Description: "{caption_text}" JSON List Output:"""
    print(f"Calling LLM Endpoint: {llm_endpoint_name}")
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": "You extract clothing items/colors as JSON list."}, {"role": "user", "content": prompt}], model=llm_endpoint_name, max_tokens=250, temperature=0.1)
        content = None; reason = "unknown"
        if chat_completion.choices and chat_completion.choices[0].message: content = chat_completion.choices[0].message.content
        if chat_completion.choices and chat_completion.choices[0].finish_reason: reason = chat_completion.choices[0].finish_reason
        print(f"LLM Raw: {content}"); print(f"LLM Finish: {reason}")
        if content is None: error_msg = "LLM returned null/empty response."; if reason == 'content_filter': error_msg += " (Content filter)"; return None, error_msg
        json_list_str = None
        try:
            start = content.find('['); end = content.rfind(']')
            if start != -1 and end != -1 and end >= start:
                json_list_str = content[start : end + 1]; print(f"Extracted JSON: {json_list_str}"); data = json.loads(json_list_str)
                if isinstance(data, list):
                    valid_list = []; found = False
                    for item in data:
                        if isinstance(item, dict) and 'color' in item and 'type' in item:
                            item['color'] = str(item.get('color', 'unknown')); item['type'] = str(item.get('type', 'unknown')); valid_list.append(item); found = True
                        else: print(f"Warn: Invalid item format: {item}")
                    if not found and data: return None, f"LLM list items lacked keys: {data}"
                    print(f"LLM Parsed: {valid_list}"); return valid_list, None
                else: return None, f"Parsed data not list. Type: {type(data)}"
            else: return None, f"Could not find JSON list markers '[]' in: '{content}'"
        except json.JSONDecodeError as e: return None, f"Failed JSON parse. Extracted: '{json_list_str}'. Raw: '{content}'. Error: {e}"
        except Exception as e: traceback.print_exc(); return None, f"Error processing LLM JSON: {e}"
    except Exception as e: error_detail = str(e); print(f"LLM API Error: {error_detail}"); traceback.print_exc(); return None, f"LLM API Call Failed: {e}"

# ---- Streamlit App User Interface ----
st.set_page_config(layout="wide", page_title="Clothing Analyzer")
st.title("👚👕👖 AI Clothing Analyzer")
st.markdown("Upload one or more images, I'll use Databricks AI (Vision + LLM) to identify color and type for each!")

# --- MODIFIED: File Uploader for Multiple Files ---
uploaded_files = st.file_uploader(
    "Choose clothing images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # Allow multiple files
)
# --- END MODIFICATION ---

# --- MODIFIED: Process Multiple Files ---
if uploaded_files: # Check if the list is not empty
    st.write(f"Processing {len(uploaded_files)} image(s)...")

    # Button to trigger analysis for ALL uploaded images
    if st.button("✨ Analyze All Uploaded Images"):
        # Iterate through each uploaded file object
        for uploaded_file in uploaded_files:
            st.divider() # Add a separator between results for each image
            image_bytes = uploaded_file.getvalue()

            # Use columns for layout for each image
            col1, col2 = st.columns([1, 2]) # Image on left, results on right

            with col1:
                st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            # Process and display results in the second column
            with col2:
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
                    st.info(f"Intermediate Caption:\n\"_{caption}_\"")

                    # --- Step 2: Call LLM endpoint ---
                    with st.spinner(f'Analyzing {uploaded_file.name}: LLM (2/2)...'):
                         final_result_list, error_message = call_llm_endpoint(caption, LLM_ENDPOINT_NAME, DATABRICKS_HOST, DATABRICKS_API_TOKEN)

                    # --- Handle LLM List Result (with color styling) ---
                    if error_message:
                         st.error(f"**LLM Analysis Failed ({uploaded_file.name}):** {error_message}")
                    elif final_result_list is not None:
                         st.subheader(f"Analysis Results ({uploaded_file.name}):")
                         if not final_result_list:
                              st.warning("LLM analysis didn't identify specific items.")
                         else:
                              for item_index, item in enumerate(final_result_list):
                                  color_str = item.get('color', 'unknown')
                                  item_type_str = item.get('type', 'unknown').capitalize()
                                  color_str_lower = color_str.lower()

                                  css_color = "inherit" # Default text color
                                  if color_str_lower != "unknown":
                                       css_color = COLOR_MAP.get(color_str_lower, "inherit")

                                  if color_str_lower != "unknown":
                                       display_string = f"<span style='color:{css_color}; font-weight:bold;'>{color_str.capitalize()}</span> {item_type_str}"
                                  else:
                                       display_string = f"{item_type_str}" # Omit 'unknown' color

                                  st.markdown(f"- {display_string}", unsafe_allow_html=True)

                         # Optionally display raw JSON list
                         # with st.expander("Show Raw LLM JSON Response"):
                         #     st.json(final_result_list)
                    else:
                         st.error("LLM Analysis failed to return a valid result or error.")
                else:
                     st.error("Vision analysis failed to return a caption or error.")

# --- Sidebar Info (Unchanged) ---
st.sidebar.header("Backend Info")
# ...(rest of sidebar code)...
st.sidebar.markdown(f"""
This app uses a two-stage AI pipeline hosted on Databricks:
1.  A custom **Vision Model** (BLIP) endpoint provides an image caption.
2.  A **Foundation Model API** (LLM) endpoint extracts details from the caption.
""")
st.sidebar.caption(f"Vision Endpoint Used: ...{BLIP_ENDPOINT_URL[-40:]}")
st.sidebar.caption(f"LLM Endpoint Name Used: {LLM_ENDPOINT_NAME}")
st.sidebar.caption(f"Databricks Host Used: {DATABRICKS_HOST}")
st.sidebar.info("Configuration read from Streamlit Secrets or environment variables.")