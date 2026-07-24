import urllib.request
import json
import urllib.error

url = "https://jarvis-backend-0cvr.onrender.com/api/model/sync"
payload = {"share_link": "https://drive.google.com/file/d/10LZDLeaQssJWGrvZM5ITBEIlv8jJff1q/view?usp=sharing"}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
)

try:
    response = urllib.request.urlopen(req)
    print("Success:", response.read().decode())
except urllib.error.HTTPError as e:
    print("Error Code:", e.code)
    print("Error Details:", e.read().decode())
except Exception as e:
    print("Exception:", e)
