## gratheon/models-bee-detector

Microservice that detects bees

- Essentially Uses Ultralytics yolov5 model
- Runs as a http server
- Dockerized
- Uses weights by Matt Nudi
  https://github.com/mattnudi/bee-detection
  https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb

## Architecture

### Service diagram

```mermaid
flowchart LR
    web-app("<a href='https://github.com/Gratheon/web-app'>web-app</a>\n:8080") --> graphql-router("<a href='https://github.com/Gratheon/graphql-router'>graphql-router</a>") --> image-splitter("<a href='https://github.com/Gratheon/image-splitter'>image-splitter</a>\n:8800") --"send cropped image"-->models-bee-detector
```


## Usage

```bash
# start service in cpu mode
just start

# alternatively, start in prod gpu mode
just start-jetson
```

### CLI usage on bare host

This assumes you have installed all of the old dependencies

```
# webcam
python detect.py --weights yolov5s.pt --source 0

# video file
python detect.py --weights yolov5s.pt --source file.mp4
```

## Ultralytics yolo v5 license

YOLOv5 is available under two different licenses:

- **GPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source requirements of GPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).

```

```
