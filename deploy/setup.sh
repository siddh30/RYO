#!/bin/bash
set -e

echo "=== RYO Server Setup ==="

# 1. System packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git

# 2. Clone repo
cd /home/ubuntu
git clone https://github.com/siddh30/RYO.git Ryo
cd Ryo

# 3. Python venv
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r deploy/requirements-server.txt

# 4. Create .env (fill in your keys after this step)
if [ ! -f .env ]; then
    cat > .env <<'EOF'
ANTHROPIC_API_KEY=""
OPENWEATHERMAP_API_KEY=""
DISCORD_TOKEN=""
EOF
    echo ""
    echo ">>> .env created — fill in your API keys before continuing <<<"
    echo "    nano /home/ubuntu/Ryo/.env"
    echo ""
    read -p "Press Enter once you've saved your keys..."
fi

# 5. Init database
venv/bin/python memory/setup_db.py

# 6. Install and start systemd service
sudo cp deploy/ryo.service /etc/systemd/system/ryo.service
sudo systemctl daemon-reload
sudo systemctl enable ryo
sudo systemctl start ryo

echo ""
echo "=== Done ==="
echo "Check status:  sudo systemctl status ryo"
echo "Live logs:     sudo journalctl -u ryo -f"
