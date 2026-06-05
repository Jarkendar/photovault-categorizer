# Deployment — Pi 5 (systemd timer)

## Prerequisites

- Docker installed and running on the Pi.
- The `photovault-categorizer` Docker image built for `linux/arm64`:
  ```bash
  docker buildx build --platform linux/arm64 -t photovault-categorizer .
  ```
- A Tailscale connection from the Pi to the machine running Postgres.

## Environment file

Create `/etc/photovault/categorizer.env` (root-readable, mode 0600):

```env
DB_URL=jdbc:postgresql://<tailscale-ip>:5432/photovault
DB_USER=photovault
DB_PASSWORD=your-db-password
PHOTO_STORAGE_ROOT=/photos
VECTOR_STORE_DIR=/vectors
PROMPTS_PATH=/app/prompts.yaml
TAG_THRESHOLD=0.25
CATEGORY_TOP_K=1
CATEGORY_MIN_SCORE=0.20
```

`PHOTO_STORAGE_ROOT` and `VECTOR_STORE_DIR` refer to paths **inside the container**;
they are bind-mounted from the Pi's filesystem via the `-v` flags in the service unit.
Adjust the host paths in the `.service` file to match your layout.

## Install and enable

```bash
sudo cp photovault-categorizer.service photovault-categorizer.timer \
        /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now photovault-categorizer.timer

# Verify timer is scheduled:
systemctl status photovault-categorizer.timer

# Run once immediately (for testing):
sudo systemctl start photovault-categorizer.service

# Watch logs:
journalctl -u photovault-categorizer.service -f
```

## n8n alternative

If you prefer n8n over systemd, create a **Cron node** that fires daily at 03:00 and
executes an **Execute Command node**:

```
/usr/bin/docker run --rm \
  --env-file /etc/photovault/categorizer.env \
  -v /srv/photovault/photos:/photos:ro \
  -v /srv/photovault/vectors:/vectors \
  photovault-categorizer
```

The lock file (`/tmp/photovault-categorize.lock`) prevents overlapping runs regardless
of the trigger mechanism, so both approaches are safe to use together.
