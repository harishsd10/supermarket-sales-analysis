import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep Render logs clean
        pass

print(f"Starting Supermarket Sales Intelligence Server on port {PORT}...")
with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
    print(f"Server is live at http://0.0.0.0:{PORT}")
    httpd.serve_forever()
