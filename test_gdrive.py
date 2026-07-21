import re
import urllib.request
import urllib.parse
import http.cookiejar
import shutil

def download_file_from_google_drive(share_link: str, destination: str):
    file_id = None
    file_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', share_link)
    if file_id_match:
        file_id = file_id_match.group(1)
    else:
        id_param_match = re.search(r'id=([a-zA-Z0-9-_]+)', share_link)
        if id_param_match:
            file_id = id_param_match.group(1)
            
    if not file_id:
        return False
        
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    try:
        # Use CookieJar to store session cookies for the confirmation bypass
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        response = opener.open(req)
        content = response.read()
        
        # Check if we got the confirmation page (contains the download form)
        html_str = content.decode('utf-8', errors='ignore')
        if 'id="download-form"' in html_str:
            print("Detected Google Drive large file warning page. Parsing form inputs...")
            
            # Extract action URL
            action_match = re.search(r'action="([^"]+)"', html_str)
            action_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
            
            # Extract all hidden inputs: name="XXX" value="YYY"
            inputs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)">', html_str)
            params = {name: val for name, val in inputs}
            
            # Construct the final download URL
            query_str = urllib.parse.urlencode(params)
            download_url = f"{action_url}?{query_str}"
            print(f"Bypassing with final download URL: {download_url}")
            
            req2 = urllib.request.Request(
                download_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            response2 = opener.open(req2)
            with open(destination, 'wb') as out_file:
                shutil.copyfileobj(response2, out_file)
            print("Successfully downloaded with confirmation bypass!")
            return True
        else:
            # If no warning page, save direct content
            with open(destination, 'wb') as out_file:
                out_file.write(content)
            print("Successfully downloaded directly!")
            return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

# Run the test
download_file_from_google_drive(
    "https://drive.google.com/file/d/1y_xx-QO6CZyhXgJNfR7ZXu2qfcyGAFr6/view?usp=sharing",
    "custom_llm/checkpoint_cloud.pt"
)
