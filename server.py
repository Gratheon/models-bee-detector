import datetime
print(f"[{datetime.datetime.now()}] Script start", flush=True)
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
# import tempfile # No longer needed
import os
import cgi
import time
# import subprocess # No longer needed for rm -rf
from detect import run # Uncommented
print(f"[{datetime.datetime.now()}] Imports finished", flush=True) # Reverted log message

# Define the request handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Handle GET requests
    def do_GET(self):
        self.send_response(200)  # Send 200 OK status code
        self.send_header("Content-type", "text/html")
        self.end_headers()

        # Send the HTML form as the response body
        form_html = """
        <html>
        <body>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit" value="Upload" />
        </form>
        </body>
        </html>
        """
        self.wfile.write(form_html.encode("utf-8"))

    # Handle POST requests
    def do_POST(self):
        content_type = self.headers["Content-Type"]

        # Remove temporary directory creation
        # reqdir = "/app/tmp/" + str(time.time()) + "/"
        # os.makedirs(reqdir, exist_ok=True)

        # Check if the content type is multipart/form-data
        if content_type.startswith("multipart/form-data"):
            # Parse the form data
            form_data = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"}
            )

            # Check if 'file' field exists
            if "file" not in form_data:
                 self.send_response(400)
                 self.send_header("Content-type", "application/json")
                 self.end_headers()
                 response = {"message": "Missing 'file' field in form data"}
                 self.wfile.write(json.dumps(response).encode("utf-8"))
                 return

            file_field = form_data["file"]

            # Check if it's a valid file upload FieldStorage instance with a filename
            if not isinstance(file_field, cgi.FieldStorage) or not file_field.filename:
                 self.send_response(400)
                 self.send_header("Content-type", "application/json")
                 self.end_headers()
                 response = {"message": "'file' field is not a valid file upload"}
                 self.wfile.write(json.dumps(response).encode("utf-8"))
                 return

            # Read file content into memory from the file-like object
            image_data = file_field.file.read()

            # Remove temporary file saving logic
            # with tempfile.NamedTemporaryFile(dir="/app/tmp", delete=False) as tmp_file:
            #     tmp_file.write(file_field.file.read())
            #     tmp_file_path = tmp_file.name
            #     filename = os.path.basename(file_field.filename)
            #     new_filename = reqdir + filename
            #     os.rename(tmp_file_path, new_filename)

            # Set weights path directly (copied to /app/weights in Dockerfile.prod)
            weights = "/app/weights/best.pt"
            device = "cpu"

            # Determine device (keep this logic, though ENV_ID check for weights is removed)
            # if os.getenv("ENV_ID") == "dev": # Removed conditional weights path
            #     weights = "/weights/best.pt"

            if os.getenv("CUDA_VISIBLE_DEVICES") != "":
                device = "cpu"

            # Call run with image_buffer, disable file saving
            detections = run(
                weights=weights,
                device=device,
                image_buffer=image_data, # Pass image data directly
                source=None,             # No file source
                project=None,            # No project dir needed
                save_txt=False,          # Disable saving txt
                nosave=True,             # Ensure no image/video save
                conf_thres=0.3,          # Keep thresholds or adjust as needed
                iou_thres=0.2,
            )

            # Process the returned detections list
            if not detections:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Nothing found", "result": []} # Return empty list
                self.wfile.write(json.dumps(response).encode("utf-8"))
                # No need for subprocess.call(["rm", "-rf", reqdir])
                return

            # Format detections for the response
            response = {"message": "File processed successfully", "result": detections}

            # No need for subprocess.call(["rm", "-rf", reqdir])

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))


# Create an HTTP server with the request handler
server_address = ("", 8700)  # Listen on all available interfaces, port 8700
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Start the server
print(f"[{datetime.datetime.now()}] Starting server...", flush=True)
print("Server running on port 8700", flush=True)
httpd.serve_forever()
