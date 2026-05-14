from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cgi
import json
import os
import time
import uuid

from gratheon_log_lib import bind_context, clear_context, configure, error_enriched, info, warn

from detect import run


configure()


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        info(
            "http access log",
            {
                "remote_addr": self.address_string(),
                "request_line": self.requestline,
                "message": format % args,
            },
        )

    def do_GET(self):
        request_id = str(uuid.uuid4())[:8]
        bind_context(request_id=request_id)
        info(
            "serving bee detector upload form",
            {
                "path": self.path,
                "method": "GET",
                "remote_addr": self.client_address[0] if self.client_address else None,
            },
        )
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

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
        clear_context()

    def do_POST(self):
        request_id = str(uuid.uuid4())[:8]
        started_at = time.perf_counter()
        bind_context(request_id=request_id)
        content_type = self.headers.get("Content-Type", "")

        info(
            "incoming bee detector request",
            {
                "path": self.path,
                "method": "POST",
                "remote_addr": self.client_address[0] if self.client_address else None,
                "content_type": content_type,
                "content_length": self.headers.get("Content-Length"),
            },
        )

        try:
            if not content_type.startswith("multipart/form-data"):
                warn("rejecting request, unsupported content type", {"content_type": content_type})
                self.send_response(415)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Unsupported content type. Please use multipart/form-data."}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            form_data = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"}
            )

            if "file" not in form_data:
                warn("rejecting request, missing file field")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Missing 'file' field in form data"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            file_field = form_data["file"]

            if not isinstance(file_field, cgi.FieldStorage) or not file_field.filename:
                warn("rejecting request, invalid file upload")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "'file' field is not a valid file upload"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            image_data = file_field.file.read()
            info(
                "uploaded bee image received",
                {
                    "filename": file_field.filename,
                    "image_bytes": len(image_data),
                },
            )

            weights = "/app/weights/best.pt"
            device = "cpu"
            if os.getenv("CUDA_VISIBLE_DEVICES") != "":
                device = "cpu"

            info(
                "starting bee detection inference",
                {
                    "weights": weights,
                    "device": device,
                    "conf_thres": 0.3,
                    "iou_thres": 0.2,
                },
            )

            detections = run(
                weights=weights,
                device=device,
                image_buffer=image_data,
                source=None,
                project=None,
                save_txt=False,
                nosave=True,
                conf_thres=0.3,
                iou_thres=0.2,
            )

            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            info(
                "bee detector request processed",
                {
                    "detections": len(detections) if detections else 0,
                    "duration_ms": duration_ms,
                },
            )

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            if not detections:
                response = {"message": "Nothing found", "result": []}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            response = {"message": "File processed successfully", "result": detections}
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            error_enriched("bee detector request failed", exc)
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Error processing image", "result": []}
            self.wfile.write(json.dumps(response).encode("utf-8"))
        finally:
            clear_context()


server_address = ("", 8700)
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

info("starting bee detector server", {"port": 8700})
httpd.serve_forever()
