import http.server
import socketserver
import os
import json
import urllib.request

PORT = int(os.environ.get("PORT", 8080))
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

class SupermarketHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/gemini-insights':
            if not GEMINI_KEY:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'GEMINI_API_KEY environment variable is not configured.'}).encode('utf-8'))
                return

            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(body) if body else {}

                context = data.get('context', 'Supermarket Sales Analytics Dataset')
                prompt = (
                    f"You are a retail executive consultant. Based on these metrics: {context}. "
                    f"Provide 3 concise, high-impact data-driven strategic actions for store management."
                )

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
                payload = {'contents': [{'parts': [{'text': prompt}]}]}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    res_data = json.loads(resp.read().decode('utf-8'))
                    reply = res_data['candidates'][0]['content']['parts'][0]['text'].strip()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'insight': reply}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    print(f"Starting Supermarket Sales Intelligence Server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), SupermarketHandler) as httpd:
        print(f"Server is live at http://0.0.0.0:{PORT}")
        httpd.serve_forever()
