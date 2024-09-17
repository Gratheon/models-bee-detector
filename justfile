start:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up --build

start-jetson:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.jetson.yml up --build

stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down

