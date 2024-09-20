cd /www/models-bee-detector/

chmod 777 tmp
chmod 755 weights

rm -rf /www/models-bee-detector/tmp/*

COMPOSE_PROJECT_NAME=gratheon docker-compose down
COMPOSE_PROJECT_NAME=gratheon docker-compose up -d --build
