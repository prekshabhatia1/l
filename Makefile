.PHONY: up down logs test

# Environment variables must be set in your shell before running 'make up'
# e.g., export WEBHOOK_SECRET="testsecret"

# make up: docker compose up -d --build [cite: 208]
up:
	@echo "Starting the application..."
	docker compose up -d --build

# make down: docker compose down -v [cite: 209]
down:
	@echo "Stopping and removing containers and volumes..."
	docker compose down -v

# make logs: docker compose logs -f api [cite: 210]
logs:
	@echo "Following API logs..."
	docker compose logs -f api

# make test: run tests (assuming you are using pytest) [cite: 211]
test:
	@echo "Running tests..."
	docker compose run --rm api pytest /app/tests

# You would still need to write the tests in the /tests directory.