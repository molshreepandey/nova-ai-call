# Deploying to AWS

The app runs on a single small Ubuntu server. One script installs everything, runs the app
as a service that restarts on failure or reboot, and puts **Caddy** in front for **HTTPS**.

> **HTTPS is required.** Browsers only allow microphone access on secure (https) pages, so
> the plain `http://<ip>` address won't work. Without a domain, the script uses
> `https://<your-ip>.sslip.io`, a free hostname that resolves to your server's IP, and Caddy
> gets a Let's Encrypt certificate for it automatically.

## What you need

- An AWS account
- Your Groq API key
- About 15 minutes

## Option A: AWS Lightsail (simplest, recommended)

1. **Create the server.** Open the [Lightsail console](https://lightsail.aws.amazon.com) → **Create instance**.
   - Platform **Linux/Unix**, blueprint **OS Only → Ubuntu 24.04 LTS**
   - Plan: **2 GB RAM** (about $12/month; check the current price on the page). The 1 GB plan also works because the script adds swap, but it is slower.
   - Give it a name, e.g. `nova-ai-call`, and click **Create instance**.
2. **Give it a fixed IP.** Open the instance → **Networking** tab → **Attach static IP**.
   Without this, the IP and the HTTPS address change whenever the instance restarts.
3. **Open the HTTPS port.** On the same **Networking** tab, under **IPv4 Firewall** → **Add rule** →
   **HTTPS (443)**. Port 80 (HTTP) is open by default; keep it, because the certificate check uses it.
4. **Connect.** Click **Connect using SSH**; a terminal opens in the browser.
5. **Install.** Paste:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/molshreepandey/nova-ai-call/main/deploy/setup.sh -o setup.sh
   sudo bash setup.sh
   ```

   It asks for your **Groq API key** (typing is hidden) and an optional **access code**.
   Set a code if the URL will be shared, so strangers can't use up your Groq quota.
   The first run takes about 5 to 10 minutes.
6. **Open the URL it prints**, e.g. `https://3-110-25-14.sslip.io`, press **Start call** and allow
   the microphone. The very first HTTPS visit can take about 30 seconds while the certificate is issued.

## Option B: Amazon EC2

1. **EC2 console → Launch instance**
   - AMI: **Ubuntu Server 24.04 LTS**
   - Instance type: **t3.small** (2 GB). `t3.micro` (free tier) works thanks to swap, but slowly.
   - Key pair: create one and download the `.pem` file
   - Network settings: allow **SSH (22)**, **HTTP (80)** and **HTTPS (443)** from anywhere
   - Storage: 16 GB
2. **Elastic IP:** EC2 → Elastic IPs → **Allocate** → **Associate** it with the instance.
3. **Connect:** `ssh -i your-key.pem ubuntu@<elastic-ip>`, or use **EC2 Instance Connect** in the console.
4. Run the same two commands as in step 5 above.

## Day-to-day

| Task | Command (on the server) |
|---|---|
| Deploy the latest code from GitHub | `sudo bash setup.sh` |
| Live logs | `sudo journalctl -u nova-agent -f` |
| Restart | `sudo systemctl restart nova-agent` |
| Change the access code | `sudo ACCESS_CODE=newcode bash setup.sh` |
| Change the Groq key | edit `/opt/nova-ai-call/.env`, then `sudo systemctl restart nova-agent` |

## Using your own domain

Create a DNS **A record** (e.g. `voice.yourdomain.com`) pointing to the static or Elastic IP, then run:

```bash
sudo DOMAIN=voice.yourdomain.com bash setup.sh
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Browser shows a certificate or "not secure" error | Check that ports 80 and 443 are open in the Lightsail firewall or EC2 security group, wait a minute, then reload. `sudo journalctl -u caddy -n 50` shows certificate errors. |
| "Could not reach the agent" | `sudo journalctl -u nova-agent -n 50`. A missing or invalid Groq key is the usual cause. |
| Microphone blocked | The page must be opened via **https://**, and the site needs mic permission (lock icon in the address bar). |
| "I'm handling a lot of requests right now" | Groq free-tier limits were reached. They reset over time; an access code limits who can use the site. |
| Slow first answer after a restart | The embedding model loads at startup (a few seconds). |

## Costs and cleanup

You pay for the server while it exists, even when idle. To stop all charges, delete the
instance **and** release the static IP (Lightsail) or Elastic IP (EC2). An unattached IP is
billed too.
