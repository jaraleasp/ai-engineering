# estimator-web

Rails 8 frontend + business backend for the AI-engineering monorepo. Consumes the FastAPI service in `../estimator/` for LLM estimations.

## Stack

- Ruby 3.4.4 / Rails 8.0.5
- PostgreSQL 16 (containerized)
- Tailwind CSS 4 via `tailwindcss-rails` (standalone binary, no Node required)
- Hotwire (Turbo + Stimulus) + Importmap + Propshaft
- Solid Cache / Queue / Cable — all in-memory in development, no Redis

## Quick start (Docker, recommended)

```bash
cp .env.example .env
docker compose up --build
```

App at http://localhost:3000. Healthcheck at `/up`.

The first build compiles native gems (`bootsnap`, `nokogiri`, `pg`, `debug`) and may take several minutes; subsequent builds are cached unless `Gemfile.lock` changes.

## Quick start (local, no Docker)

Requires Postgres listening on the local Unix socket.

```bash
bin/setup        # bundle install + db:prepare
bin/dev          # Puma + tailwindcss:watch via foreman, port 3000
```

`config/database.yml` reads `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD` from ENV. When unset, Rails falls back to the Unix socket — that is the local-without-docker path.

## Common commands

```bash
docker compose exec estimator-web bin/rails console
docker compose exec estimator-web bin/rails db:migrate
docker compose exec estimator-web bin/rails test
docker compose exec estimator-web bash

# Connect to the dev DB with psql
docker compose exec postgres psql -U postgres estimator_web_development

# Live logs
docker compose logs -f estimator-web

# Add a gem (rebuild to bake it into the image)
docker compose exec estimator-web bundle add <gem>
docker compose build estimator-web
```

## Cross-service calls

When this project is launched from the **monorepo root** (`docker compose up` in `../`), it shares a network with the FastAPI estimator. Rails can reach it at `http://estimator:8000` (see `ESTIMATOR_API_BASE_URL` in `.env.example`). Launching `estimator-web` standalone leaves that hostname unresolvable — by design.

## Production / Kamal

Out of scope. The `Dockerfile` here is development-only. The `kamal` and `thruster` gems and the `.kamal/` + `config/deploy.yml` files are leftovers from `rails new` and currently unused; they do not load at runtime (`require: false`) and can be removed if production stays off the table.
